# macOS Re-build Instructions for Whisper Library Fix

## Problem
The app crashes on startup with:
```
Library not loaded: @rpath/libwhisper.1.dylib
```

## Solution Applied
The fix has been committed to branch `claude/session-011CUZLKpPMJGUKtxFxDp39w`. Now you need to rebuild the app on macOS.

## Step 1: Pull the Latest Changes

```bash
git fetch origin
git checkout claude/session-011CUZLKpPMJGUKtxFxDp39w
git pull origin claude/session-011CUZLKpPMJGUKtxFxDp39w
```

## Step 2: Clean Build Directory

It's important to clean the build cache to ensure the new CMake configuration is applied:

```bash
# If you have a build directory
rm -rf build/
mkdir build
cd build
```

## Step 3: Configure with CMake

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

Or if you use a specific generator:
```bash
cmake .. -G "Xcode" -DCMAKE_BUILD_TYPE=Release
```

## Step 4: Build the Application

```bash
cmake --build . --config Release
```

Or if using Xcode:
```bash
xcodebuild -configuration Release
```

## Step 5: Verify the Fix

After building, verify that the library is properly bundled:

```bash
# Check if Frameworks directory exists
ls -la ShioRIS3.app/Contents/Frameworks/

# Should show libwhisper.1.dylib (or similar whisper library file)

# Verify the library reference
otool -L ShioRIS3.app/Contents/MacOS/ShioRIS3 | grep whisper

# Should show: @rpath/libwhisper.1.dylib
```

## Step 6: Test the Application

```bash
# Run the application
open ShioRIS3.app
```

## What the Fix Does

The CMake configuration now:

1. **Adds RPATH**: `@executable_path/../Frameworks` is added to the runtime library search path
2. **Copies Library**: POST_BUILD command copies `libwhisper.1.dylib` to `ShioRIS3.app/Contents/Frameworks/`
3. **Fixes Install Name**: Uses `install_name_tool` to set proper library references

## Troubleshooting

### If the library is still not found:

Check the actual library name:
```bash
ls -la build/external/whisper.cpp/
```

If the library has a different name (e.g., `libwhisper.dylib` without version), you may need to adjust the CMake script.

### If POST_BUILD commands didn't run:

```bash
# Manually copy the library
mkdir -p ShioRIS3.app/Contents/Frameworks
cp build/external/whisper.cpp/libwhisper*.dylib ShioRIS3.app/Contents/Frameworks/

# Fix install name
install_name_tool -id "@rpath/libwhisper.1.dylib" \
    ShioRIS3.app/Contents/Frameworks/libwhisper.1.dylib

# Fix reference in executable
install_name_tool -change \
    /path/to/old/libwhisper.1.dylib \
    @rpath/libwhisper.1.dylib \
    ShioRIS3.app/Contents/MacOS/ShioRIS3
```

### Check what libraries are being referenced:

```bash
otool -L ShioRIS3.app/Contents/MacOS/ShioRIS3
```

## Expected Output

After successful build, you should see:
- No crash on startup
- Whisper voice input features working correctly
- Library properly loaded from the app bundle
