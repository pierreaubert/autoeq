//! Artifact storage contracts for AutoEQ reports, exports, and sidecars.

use autoeq_core::AutoeqError;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use tempfile::{Builder, NamedTempFile, TempDir};

/// Deterministic RoomEQ convolution-sidecar naming and reservation helpers.
pub mod roomeq;

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

/// Validate a portable relative path for an artifact bundle member.
///
/// Paths may contain portable nested components only. Absolute paths, parent
/// traversal, Windows-invalid characters, trailing dots/spaces, device names,
/// and drive prefixes are rejected on every platform so an artifact stays
/// inside its bundle when moved between systems.
///
/// # Errors
/// Returns InvalidInput when the path is not a safe portable relative path.
pub fn validate_relative_artifact_path(path: &Path) -> io::Result<()> {
    let Some(path_text) = path.to_str() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "artifact path is not valid UTF-8",
        ));
    };
    if path_text.is_empty()
        || path_text.starts_with('/')
        || path_text
            .split('/')
            .any(|component| !is_portable_artifact_component(component))
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "artifact member '{}' must be a safe relative path",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn is_portable_artifact_component(component: &str) -> bool {
    if component.is_empty()
        || component == "."
        || component == ".."
        || component.ends_with([' ', '.'])
        || component
            .chars()
            .any(|character| character.is_control() || r#"\/:<>|?*\""#.contains(character))
    {
        return false;
    }

    let device_name = component
        .split('.')
        .next()
        .unwrap_or_default()
        .trim_end_matches([' ', '.'])
        .to_ascii_lowercase();
    if matches!(device_name.as_str(), "con" | "prn" | "aux" | "nul") {
        return false;
    }

    let reserved_numbered_device = ["com", "lpt"].iter().any(|prefix| {
        let suffix = device_name.strip_prefix(prefix).unwrap_or_default();
        matches!(
            suffix,
            "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
        )
    });
    !reserved_numbered_device
}

/// Join a validated relative artifact path to its bundle root.
///
/// # Errors
/// Returns InvalidInput when the relative path contains unsafe components.
pub fn safe_artifact_path(root: &Path, relative_path: &Path) -> io::Result<PathBuf> {
    validate_relative_artifact_path(relative_path)?;
    Ok(root.join(relative_path))
}

/// Create and retain a unique directory for staging one artifact bundle.
///
/// The directory is removed automatically unless keep is called after the
/// publication commit point.
///
/// # Errors
/// Returns an I/O error when the parent cannot be created or staging cannot
/// be reserved.
#[derive(Debug)]
pub struct ArtifactBundleStaging {
    directory: TempDir,
}

impl ArtifactBundleStaging {
    /// Create a unique staging directory inside the given parent.
    ///
    /// # Errors
    /// Returns an I/O error when the parent cannot be created or the unique
    /// staging directory cannot be reserved.
    pub fn new(parent: impl AsRef<Path>) -> io::Result<Self> {
        let parent = parent.as_ref();
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        std::fs::create_dir_all(parent)?;
        let directory = Builder::new()
            .prefix(".autoeq-bundle-")
            .tempdir_in(parent)?;
        Ok(Self { directory })
    }

    /// Return the root directory for staged bundle members.
    pub fn root(&self) -> &Path {
        self.directory.path()
    }

    /// Resolve a safe member path and create its parent directories.
    ///
    /// # Errors
    /// Returns InvalidInput for unsafe paths or an I/O error when a parent
    /// directory cannot be created.
    pub fn member_path(&self, relative_path: &Path) -> io::Result<PathBuf> {
        let path = safe_artifact_path(self.root(), relative_path)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(path)
    }

    /// Disarm automatic cleanup and keep the directory on disk for publication
    /// or crash recovery.
    pub fn keep(self) -> PathBuf {
        self.directory.keep()
    }
}

/// Atomically replace one file without deleting its current contents first.
///
/// The temporary file is created beside the destination, so persistence stays
/// on the same filesystem. A failed persist leaves the prior destination
/// untouched.
///
/// # Errors
/// Returns an I/O error when the temporary file cannot be written or
/// atomically persisted.
pub fn write_file_atomically(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = NamedTempFile::new_in(parent)?;
    temporary.write_all(contents)?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

/// Return the SHA-256 digest of artifact bytes as lowercase hexadecimal.
pub fn sha256_hex(contents: &[u8]) -> String {
    Sha256::digest(contents)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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
        write_file_atomically(path, contents).map_err(|e| AutoeqError::FileOperation {
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
    fn relative_artifact_paths_reject_cross_platform_traversal() {
        for unsafe_path in [
            "../outside.txt",
            r"..\outside.txt",
            r"C:\outside.txt",
            "/outside.txt",
            "nested//empty.txt",
            "nested/name?.txt",
            "nested/name*.txt",
            "nested/name<.txt",
            "nested/name>.txt",
            "nested/name|.txt",
            "nested/name\".txt",
            "nested/trailing-dot.",
            "nested/trailing-space ",
            "CON",
            "aux.txt",
            "COM1.csv",
            "LPT9.csv",
            "COM¹.csv",
        ] {
            assert!(
                super::validate_relative_artifact_path(Path::new(unsafe_path)).is_err(),
                "accepted unsafe path {unsafe_path:?}"
            );
        }
        assert!(super::validate_relative_artifact_path(Path::new("nested/inside.txt")).is_ok());
    }

    #[test]
    fn atomic_write_preserves_destination_when_persist_fails() {
        let tmp = tempfile::TempDir::new().unwrap();
        let destination = tmp.path().join("existing-directory");
        std::fs::create_dir(&destination).unwrap();
        let marker = destination.join("marker.txt");
        std::fs::write(&marker, b"old bundle").unwrap();

        assert!(super::write_file_atomically(&destination, b"replacement").is_err());
        assert_eq!(std::fs::read(marker).unwrap(), b"old bundle");
    }

    #[test]
    fn staging_directories_are_unique_and_cleaned_on_drop() {
        let tmp = tempfile::TempDir::new().unwrap();
        let first = super::ArtifactBundleStaging::new(tmp.path()).unwrap();
        let second = super::ArtifactBundleStaging::new(tmp.path()).unwrap();
        let first_path = first.root().to_path_buf();
        let second_path = second.root().to_path_buf();
        assert_ne!(first_path, second_path);
        let member = first.member_path(Path::new("nested/member.txt")).unwrap();
        std::fs::write(member, b"contents").unwrap();

        drop(first);
        assert!(!first_path.exists());
        assert!(second_path.is_dir());
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
