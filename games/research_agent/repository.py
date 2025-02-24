"""Repository management system for Research Agent."""

import shutil
import difflib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import hashlib

class RepositoryManager:
    """Manages a versioned repository of files with history tracking."""
    
    def __init__(self, repo_dir: Path):
        """Initialize repository structure.
        
        Args:
            repo_dir: Base directory for the repository
        """
        self.repo_dir = Path(repo_dir)
        self.working_dir = self.repo_dir / "repository"  # Create a repository subdirectory
        self.history_dir = self.repo_dir / ".history"  # Hide history in .history
        self.logs_dir = self.repo_dir / ".logs"  # Hide logs in .logs
        self.index_file = self.repo_dir / ".index.json"  # Hide index in .index.json
        
        # Create directory structure
        for directory in [self.working_dir, self.history_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize or load index
        self._init_index()
        
    def _init_index(self) -> None:
        """Initialize or load the repository index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "files": {},  # path -> {versions: []}
                "latest_version": 0
            }
            self._save_index()
            
    def _save_index(self) -> None:
        """Save the current index state."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
            
    def _get_file_hash(self, content: str) -> str:
        """Generate hash for file content."""
        return hashlib.sha256(content.encode()).hexdigest()
        
    def _log_operation(self, operation: str, path: str, details: Dict[str, Any]) -> None:
        """Log repository operations with timestamp."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "path": str(path),
            "details": details
        }
        
        log_file = self.logs_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def ls(self, path: str = ".") -> List[Dict[str, Any]]:
        """List contents of directory with metadata.
        
        Args:
            path: Relative path to list, defaults to root
            
        Returns:
            List of file/directory entries with metadata
            
        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If path is outside working directory
        """
        # Convert path to absolute and resolve any symlinks
        target = (self.working_dir / path).resolve()
        working_dir_resolved = self.working_dir.resolve()
        
        if not target.exists():
            error_msg = f"Path does not exist: {path}"
            print(f"Error: {error_msg}")
            print(f"Target path: {target}")
            print(f"Working directory: {working_dir_resolved}")
            raise FileNotFoundError(error_msg)
            
        # Ensure we don't escape working directory by checking if target is a subpath
        if not str(target).startswith(str(working_dir_resolved)):
            error_msg = f"Access denied: Path '{path}' is outside working directory"
            print(f"Error: {error_msg}")
            print(f"Target path: {target}")
            print(f"Working directory: {working_dir_resolved}")
            raise ValueError(error_msg)
            
        results = []
        try:
            for item in target.iterdir():
                stats = item.stat()
                entry = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": stats.st_size,
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
                }
                
                # Add version info for files
                rel_path = str(item.relative_to(self.working_dir))
                if entry["type"] == "file" and rel_path in self.index["files"]:
                    entry["versions"] = len(self.index["files"][rel_path]["versions"])
                    
                results.append(entry)
                
            self._log_operation("ls", path, {"entries": len(results)})
            return results
            
        except Exception as e:
            error_msg = f"Error listing directory '{path}': {str(e)}"
            print(f"Error: {error_msg}")
            print(f"Target path: {target}")
            print(f"Working directory: {working_dir_resolved}")
            self._log_operation("ls_error", path, {"error": str(e)})
            raise

    def read_file(self, path: str, version: Optional[int] = None) -> str:
        """Read contents of a file.
        
        Args:
            path: Path to file relative to working directory
            version: Optional specific version to read
            
        Returns:
            File contents as string
        """
        if version is not None:
            # Read from history
            if path not in self.index["files"]:
                raise FileNotFoundError(f"No history for file: {path}")
                
            versions = self.index["files"][path]["versions"]
            if version >= len(versions):
                raise ValueError(f"Version {version} not found for {path}")
                
            version_info = versions[version]
            version_file = self.history_dir / str(version) / path
            
            if not version_file.exists():
                raise FileNotFoundError(f"Version file missing: {version_file}")
                
            content = version_file.read_text()
            
            # Verify hash
            if self._get_file_hash(content) != version_info["hash"]:
                raise ValueError(f"Version {version} of {path} is corrupted")
                
            self._log_operation("read", path, {"version": version})
            return content
            
        else:
            # Read current version
            target = (self.working_dir / path).resolve()
            
            # Security check
            if not str(target).startswith(str(self.working_dir)):
                raise ValueError("Access denied: Path outside working directory")
                
            if not target.exists():
                raise FileNotFoundError(f"File not found: {path}")
                
            if not target.is_file():
                raise ValueError(f"Not a file: {path}")
                
            content = target.read_text()
            self._log_operation("read", path, {"version": "current"})
            return content

    def write_file(self, path: str, content: str) -> int:
        """Write content to a file and create new version.
        
        Args:
            path: Path to file relative to working directory
            content: Content to write
            
        Returns:
            New version number
            
        Raises:
            ValueError: If path is outside working directory
            OSError: If there are permission issues or disk space issues
        """
        rel_path = str(Path(path))
        target = (self.working_dir / rel_path).resolve()
        
        # Security check
        if not str(target).startswith(str(self.working_dir)):
            error_msg = f"Access denied: Path '{path}' is outside working directory '{self.working_dir}'"
            print(f"Error: {error_msg}")
            print(f"Target path: {target}")
            print(f"Working directory: {self.working_dir}")
            raise ValueError(error_msg)
            
        try:
            # Create parent directories
            target.parent.mkdir(parents=True, exist_ok=True)
            
        except OSError as e:
            error_msg = f"Failed to create directory structure for '{path}': {str(e)}"
            print(f"Error: {error_msg}")
            print(f"Target directory: {target.parent}")
            raise OSError(error_msg) from e
        
        # Calculate hash
        content_hash = self._get_file_hash(content)
        
        # Initialize file entry if new
        if rel_path not in self.index["files"]:
            self.index["files"][rel_path] = {"versions": []}
            
        try:
            # Check if content changed
            if target.exists():
                current_content = target.read_text()
                if self._get_file_hash(current_content) == content_hash:
                    return len(self.index["files"][rel_path]["versions"]) - 1
                    
            # Create new version
            version = self.index["latest_version"] + 1
            self.index["latest_version"] = version
            
            # Save to history
            version_dir = self.history_dir / str(version)
            version_dir.mkdir(parents=True, exist_ok=True)
            version_file = version_dir / rel_path
            version_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                version_file.write_text(content)
            except OSError as e:
                error_msg = f"Failed to write version file '{version_file}': {str(e)}"
                print(f"Error: {error_msg}")
                raise OSError(error_msg) from e
            
            # Update index
            version_info = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "hash": content_hash
            }
            self.index["files"][rel_path]["versions"].append(version_info)
            
            # Update working copy
            try:
                target.write_text(content)
            except OSError as e:
                error_msg = f"Failed to write working copy '{target}': {str(e)}"
                print(f"Error: {error_msg}")
                # Try to roll back version file
                try:
                    version_file.unlink()
                except:
                    pass
                raise OSError(error_msg) from e
            
            # Save index
            try:
                self._save_index()
            except OSError as e:
                error_msg = f"Failed to save index: {str(e)}"
                print(f"Error: {error_msg}")
                # Try to roll back changes
                try:
                    version_file.unlink()
                    target.unlink()
                except:
                    pass
                raise OSError(error_msg) from e
            
            self._log_operation("write", rel_path, {
                "version": version,
                "hash": content_hash
            })
            
            return version
            
        except Exception as e:
            error_msg = f"Unexpected error writing file '{path}': {str(e)}"
            print(f"Error: {error_msg}")
            raise

    def diff(self, path: str, version1: Optional[int], version2: Optional[int] = None) -> str:
        """Get diff between two versions of a file.
        
        Args:
            path: Path to file
            version1: First version to compare
            version2: Second version (defaults to current)
            
        Returns:
            Unified diff string
        """
        # Get content for version1
        content1 = self.read_file(path, version1) if version1 is not None else ""
        
        # Get content for version2
        content2 = self.read_file(path, version2) if version2 is not None else self.read_file(path)
        
        # Generate diff
        diff = difflib.unified_diff(
            content1.splitlines(keepends=True),
            content2.splitlines(keepends=True),
            fromfile=f"{path}@{version1 if version1 is not None else 'empty'}",
            tofile=f"{path}@{version2 if version2 is not None else 'current'}"
        )
        
        self._log_operation("diff", path, {
            "version1": version1,
            "version2": version2
        })
        
        return "".join(diff)

    def get_history(self, path: str) -> List[Dict[str, Any]]:
        """Get version history for a file.
        
        Args:
            path: Path to file
            
        Returns:
            List of version entries
        """
        if path not in self.index["files"]:
            raise FileNotFoundError(f"No history for file: {path}")
            
        return self.index["files"][path]["versions"]

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent operation logs.
        
        Args:
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entries
        """
        logs = []
        log_files = sorted(self.logs_dir.glob("*.jsonl"), reverse=True)
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    logs.append(json.loads(line))
                    if len(logs) >= limit:
                        break
            if len(logs) >= limit:
                break
                
        return logs[:limit]
