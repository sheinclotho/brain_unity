# Changelog

All notable changes to the TwinBrain framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Created comprehensive CODE_REVIEW_REPORT.md with detailed analysis
- Added requirements.txt for dependency management
- Added .gitignore for version control
- Created README.md with project overview and quick start guide
- Added configuration management system (unity_integration/config.py)
- Added input validation utilities (unity_integration/validation.py)
- Created example config.json file
- Added validation to model_server.py for safer input handling

### Changed
- Improved model_server.py with input validation
- Enhanced error handling in core modules
- Updated documentation with better examples

### Security
- Changed default WebSocket host from 0.0.0.0 to 127.0.0.1 for better security
- Added path validation to prevent path traversal attacks
- Added input sanitization for filenames
- Added validation for all user inputs

## [4.1.0] - 2024-02-15

### Added
- Automated Unity setup tools
- Virtual stimulation UI
- File monitoring and auto-reload
- One-click installation script
- Support for multiple stimulation patterns (sine, pulse, ramp, constant)

### Changed
- Improved WebSocket communication
- Enhanced JSON export format to v2.0
- Optimized memory usage in brain_state_exporter

### Fixed
- File path handling on Windows
- Unity package installation issues
- FreeSurfer data loading edge cases

## [4.0.0] - 2024-01-XX

### Added
- Complete rewrite of Unity integration layer
- WebSocket server for real-time communication
- Stimulation simulator with multiple patterns
- FreeSurfer data loader and processor
- OBJ model generator
- Brain state exporter to JSON format

### Changed
- Migrated from HTTP to WebSocket protocol
- Improved data structures for brain states
- Enhanced documentation (Chinese and English)

### Deprecated
- Old HTTP-based API

## [3.x.x] - Previous versions

Older versions (before 4.0.0) had different architectures and are not documented here.
Please refer to git history for details.

---

## Notes

### Types of Changes
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

### Migration Guides

#### From 4.0 to 4.1
- Update Unity scripts to use new WebSocketClient_Improved
- Update JSON parsing to handle v2.0 format
- Reconfigure file monitoring paths if customized

#### From 3.x to 4.0
- Complete reinstallation required
- New Unity project setup
- Update all Python dependencies
- Review and update configuration files
