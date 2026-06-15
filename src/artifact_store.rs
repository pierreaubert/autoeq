//! Artifact storage abstraction.
//!
//! The [`ArtifactStore`] trait decouples domain code from direct `std::fs`
//! calls, making it possible to run tests against an in-memory store instead
//! of the filesystem.

use crate::error::AutoeqError;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Abstraction over artifact persistence (directories, JSON exports, FIR WAVs,
/// reports, sidecars, etc.).
pub trait ArtifactStore: Send + Sync {
    /// Ensure that `path` and all its parent directories exist.
    fn create_dir_all(&self, path: &Path) -> Result<(), AutoeqError>;

    /// Write `contents` to `path`, creating or overwriting the file.
    fn write(&self, path: &Path, contents: &[u8]) -> Result<(), AutoeqError>;

    /// Read the contents of `path` if it exists.
    fn read(&self, path: &Path) -> Result<Option<Vec<u8>>, AutoeqError>;
}

/// Production implementation backed by the local filesystem.
#[derive(Debug, Default, Clone, Copy)]
pub struct FsArtifactStore;

impl FsArtifactStore {
    /// Create a new filesystem-backed store.
    pub fn new() -> Self {
        Self
    }
}

impl ArtifactStore for FsArtifactStore {
    fn create_dir_all(&self, path: &Path) -> Result<(), AutoeqError> {
        std::fs::create_dir_all(path).map_err(|e| AutoeqError::DirectoryCreation {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    fn write(&self, path: &Path, contents: &[u8]) -> Result<(), AutoeqError> {
        std::fs::write(path, contents).map_err(|e| AutoeqError::FileOperation {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    fn read(&self, path: &Path) -> Result<Option<Vec<u8>>, AutoeqError> {
        match std::fs::read(path) {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(AutoeqError::FileOperation {
                path: path.display().to_string(),
                message: e.to_string(),
            }),
        }
    }
}

/// In-memory implementation for deterministic unit tests.
#[derive(Debug, Default, Clone)]
pub struct MemoryArtifactStore {
    files: Arc<Mutex<HashMap<PathBuf, Vec<u8>>>>,
    dirs: Arc<Mutex<Vec<PathBuf>>>,
}

impl MemoryArtifactStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the bytes stored under `path`, if any.
    pub fn get(&self, path: &Path) -> Option<Vec<u8>> {
        self.files.lock().unwrap().get(path).cloned()
    }

    /// Return true if `path` has been created as a directory.
    pub fn is_dir(&self, path: &Path) -> bool {
        self.dirs.lock().unwrap().contains(&path.to_path_buf())
    }

    /// Return the number of stored files.
    pub fn file_count(&self) -> usize {
        self.files.lock().unwrap().len()
    }
}

impl ArtifactStore for MemoryArtifactStore {
    fn create_dir_all(&self, path: &Path) -> Result<(), AutoeqError> {
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

    fn write(&self, path: &Path, contents: &[u8]) -> Result<(), AutoeqError> {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_path_buf(), contents.to_vec());
        Ok(())
    }

    fn read(&self, path: &Path) -> Result<Option<Vec<u8>>, AutoeqError> {
        Ok(self.files.lock().unwrap().get(path).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::{ArtifactStore, FsArtifactStore, MemoryArtifactStore};
    use std::path::Path;

    #[test]
    fn memory_store_round_trip() {
        let store = MemoryArtifactStore::new();
        store.create_dir_all(Path::new("a/b")).unwrap();
        store.write(Path::new("a/b/c.txt"), b"hello").unwrap();
        assert!(store.is_dir(Path::new("a/b")));
        assert_eq!(store.get(Path::new("a/b/c.txt")).unwrap(), b"hello");
    }

    #[test]
    fn memory_store_missing_read_returns_none() {
        let store = MemoryArtifactStore::new();
        assert!(store.read(Path::new("missing.txt")).unwrap().is_none());
    }

    #[test]
    fn fs_store_round_trip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FsArtifactStore::new();
        let path = tmp.path().join("sub/dir/file.txt");
        store.create_dir_all(path.parent().unwrap()).unwrap();
        store.write(&path, b"world").unwrap();
        assert_eq!(store.read(&path).unwrap().unwrap(), b"world");
    }
}
