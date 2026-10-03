use std::fs::File;
use std::io;
use std::path::Path;

pub(crate) fn replace_atomically(
    path: &Path,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    write(temporary.as_file_mut())?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    sync_parent_directory(path)
}

pub(crate) fn sync_parent_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn failed_write_keeps_the_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        std::fs::write(&path, b"old complete contents").unwrap();

        let result = replace_atomically(&path, |file| {
            file.write_all(b"partial new contents")?;
            Err(io::Error::other("injected write failure"))
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"old complete contents");
    }
}
