# ShioRIS3

This project is a prototype DICOM viewer written in C++ using Qt6, DCMTK and OpenCV.

## Build Instructions

### Windows
1. Install **Visual Studio 2022** with the C++ workload.
2. Install **Qt6** (e.g. `C:\Qt\6.9.1\msvc2022_64`).
3. Install **vcpkg** and build the required libraries:
   ```powershell
   vcpkg install dcmtk opencv
   ```
4. Set the following environment variables before running CMake:
   ```powershell
   $env:Qt6_DIR = 'C:\Qt\6.9.1\msvc2022_64\lib\cmake\Qt6'
   $env:VCPKG_ROOT = 'C:\path\to\vcpkg'
   ```
5. From the project root, execute the provided build script:
   ```powershell
   .\build_windows_en.ps1 -Configuration Release
   ```
   Add `-Run` to start the application after a successful build.

### Linux
1. Install Qt6, DCMTK and OpenCV using your package manager. Example for Ubuntu:
   ```bash
   sudo apt-get install build-essential qt6-base-dev libdcmtk-dev libopencv-dev
   ```
2. Configure and build with CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make -j$(nproc)
   ```
3. Run the viewer:
   ```bash
   ./ShioRIS3
   ```

After launching the program, you can open a single DICOM file with **File → Open DICOM File...** or load an entire directory of images via **File → Open DICOM Folder...**. RT Dose files can be loaded with **File → Open RT Dose...**. When an RT Dose is loaded together with a CT volume the dose data is automatically resampled to the CT resolution to enable fast and memory‑efficient overlay display. Use the arrow keys to navigate between images.
