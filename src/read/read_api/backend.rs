//! Pluggable measurement backend and cache for `read_api`.
//!
//! This module introduces two seams that were previously hard-wired inside
//! `fetch.rs`:
//!
//! - [`MeasurementBackend`] performs the actual network (or mock) request.
//! - [`MeasurementCache`] persists and loads cached responses.
//!
//! Production implementations (`ReqwestMeasurementBackend`,
//! `FsMeasurementCache`) are used by the public free functions by default.
//! Tests can swap in an in-memory cache and a mock backend to run fully
//! offline and deterministically.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Error type alias for backend/cache operations.
pub type BackendResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Abstraction over the network/source that provides measurement data.
#[async_trait]
pub trait MeasurementBackend: Send + Sync {
    /// Fetch a URL and return the raw response body as text.
    async fn get_text(&self, url: &str) -> BackendResult<String>;
}

/// Abstraction over the local measurement cache.
#[async_trait]
pub trait MeasurementCache: Send + Sync {
    /// Read a cached file as a UTF-8 string.
    async fn read_to_string(&self, path: &Path) -> BackendResult<Option<String>>;

    /// Write a UTF-8 string to the cache.
    async fn write(&self, path: &Path, content: &str) -> BackendResult<()>;

    /// Ensure that `path` and its parents exist in the cache.
    async fn create_dir_all(&self, path: &Path) -> BackendResult<()>;
}

/// Production backend that issues real HTTP GET requests via `reqwest`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReqwestMeasurementBackend;

impl ReqwestMeasurementBackend {
    /// Create a new production backend.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MeasurementBackend for ReqwestMeasurementBackend {
    async fn get_text(&self, url: &str) -> BackendResult<String> {
        let response = reqwest::get(url).await?;
        if !response.status().is_success() {
            return Err(format!("API request failed with status: {}", response.status()).into());
        }
        Ok(response.text().await?)
    }
}

/// Production cache backed by the local filesystem.
#[derive(Debug, Default, Clone, Copy)]
pub struct FsMeasurementCache;

impl FsMeasurementCache {
    /// Create a new filesystem-backed cache.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MeasurementCache for FsMeasurementCache {
    async fn read_to_string(&self, path: &Path) -> BackendResult<Option<String>> {
        match tokio::fs::read_to_string(path).await {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn write(&self, path: &Path, content: &str) -> BackendResult<()> {
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    async fn create_dir_all(&self, path: &Path) -> BackendResult<()> {
        tokio::fs::create_dir_all(path).await?;
        Ok(())
    }
}

/// In-memory cache for deterministic tests.
#[derive(Debug, Default, Clone)]
pub struct InMemoryMeasurementCache {
    files: Arc<Mutex<HashMap<PathBuf, String>>>,
    dirs: Arc<Mutex<Vec<PathBuf>>>,
}

impl InMemoryMeasurementCache {
    /// Create a new empty in-memory cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-seed a cache entry.
    pub fn insert(&self, path: impl Into<PathBuf>, content: impl Into<String>) {
        self.files.lock().unwrap().insert(path.into(), content.into());
    }

    /// Return the content stored under `path`, if any.
    pub fn get(&self, path: &Path) -> Option<String> {
        self.files.lock().unwrap().get(path).cloned()
    }

    /// Return true if `path` has been created as a directory.
    pub fn is_dir(&self, path: &Path) -> bool {
        self.dirs.lock().unwrap().contains(&path.to_path_buf())
    }
}

#[async_trait]
impl MeasurementCache for InMemoryMeasurementCache {
    async fn read_to_string(&self, path: &Path) -> BackendResult<Option<String>> {
        Ok(self.files.lock().unwrap().get(path).cloned())
    }

    async fn write(&self, path: &Path, content: &str) -> BackendResult<()> {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_path_buf(), content.to_string());
        Ok(())
    }

    async fn create_dir_all(&self, path: &Path) -> BackendResult<()> {
        let mut dirs = self.dirs.lock().unwrap();
        let mut current = Some(path);
        while let Some(p) = current {
            let owned = p.to_path_buf();
            if !dirs.contains(&owned) {
                dirs.push(owned);
            }
            current = p.parent();
        }
        Ok(())
    }
}

/// Simple mock backend that returns a fixed response for a given URL.
#[derive(Debug, Default, Clone)]
pub struct MockMeasurementBackend {
    response: String,
}

impl MockMeasurementBackend {
    /// Create a backend that always returns `response`.
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

#[async_trait]
impl MeasurementBackend for MockMeasurementBackend {
    async fn get_text(&self, _url: &str) -> BackendResult<String> {
        Ok(self.response.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FsMeasurementCache, InMemoryMeasurementCache, MeasurementBackend, MeasurementCache,
        MockMeasurementBackend,
    };

    #[tokio::test]
    async fn in_memory_cache_round_trip() {
        let cache = InMemoryMeasurementCache::new();
        let path = std::path::Path::new("speakers/test/CEA2034.json");
        cache.create_dir_all(path.parent().unwrap()).await.unwrap();
        cache.write(path, "{\"data\":[]}").await.unwrap();
        assert!(cache.is_dir(path.parent().unwrap()));
        assert_eq!(cache.get(path).unwrap(), "{\"data\":[]}");
    }

    #[tokio::test]
    async fn mock_backend_returns_fixed_response() {
        let backend = MockMeasurementBackend::new("hello");
        assert_eq!(backend.get_text("any-url").await.unwrap(), "hello");
    }

    #[tokio::test]
    async fn fs_cache_round_trip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = FsMeasurementCache::new();
        let path = tmp.path().join("a/b/c.json");
        cache.create_dir_all(path.parent().unwrap()).await.unwrap();
        cache.write(&path, "{}").await.unwrap();
        assert_eq!(cache.read_to_string(&path).await.unwrap().unwrap(), "{}");
    }
}
