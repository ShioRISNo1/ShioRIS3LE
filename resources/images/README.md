# ShioRIS3 Image Resources

## Logo Placement

Place your logo file here: `ShioRIS3_Logo.png`

### Requirements:
- **File name**: `ShioRIS3_Logo.png` (exactly)
- **Location**: `/resources/images/ShioRIS3_Logo.png`
- **Format**: PNG (recommended for transparency support)
- **Recommended size**: 400x200 pixels or similar aspect ratio
- The logo will be automatically scaled to fit in the license dialog

### Usage:
The logo is displayed in the License Information dialog that appears when the application starts.
It is loaded via Qt Resource System with the path `:/images/ShioRIS3_Logo.png`.

### Adding the Logo:
1. Save your `ShioRIS3_Logo.png` file in this directory
2. Rebuild the project (the logo is embedded in the executable via Qt resources)
3. The logo will appear at the top of the license dialog

### Notes:
- If the logo file is not found, the dialog will simply skip displaying it (no error)
- The logo is automatically scaled to maximum width of 200 pixels while maintaining aspect ratio
- Smooth transformation is used for high-quality scaling
