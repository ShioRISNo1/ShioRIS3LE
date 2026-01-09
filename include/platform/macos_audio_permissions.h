#ifndef MACOS_AUDIO_PERMISSIONS_H
#define MACOS_AUDIO_PERMISSIONS_H

#include <QString>
#include <functional>

namespace MacOSAudioPermissions {

enum class PermissionStatus {
    Authorized,
    Denied,
    Restricted,
    NotDetermined
};

// Check the current microphone permission status
PermissionStatus checkMicrophonePermission();

// Request microphone permission (async)
// Callback is called on main thread with the result
void requestMicrophonePermission(std::function<void(bool granted)> callback);

// Get a user-friendly description of the permission status
QString permissionStatusDescription(PermissionStatus status);

} // namespace MacOSAudioPermissions

#endif // MACOS_AUDIO_PERMISSIONS_H
