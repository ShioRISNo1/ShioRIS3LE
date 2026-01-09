#include "platform/macos_audio_permissions.h"

#ifdef Q_OS_MACOS

#import <AVFoundation/AVFoundation.h>
#include <QDebug>

namespace MacOSAudioPermissions {

PermissionStatus checkMicrophonePermission()
{
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];

    switch (status) {
        case AVAuthorizationStatusAuthorized:
            return PermissionStatus::Authorized;
        case AVAuthorizationStatusDenied:
            return PermissionStatus::Denied;
        case AVAuthorizationStatusRestricted:
            return PermissionStatus::Restricted;
        case AVAuthorizationStatusNotDetermined:
            return PermissionStatus::NotDetermined;
        default:
            qWarning() << "Unknown AVAuthorizationStatus:" << status;
            return PermissionStatus::NotDetermined;
    }
}

void requestMicrophonePermission(std::function<void(bool granted)> callback)
{
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
        // Call the callback on the main thread
        dispatch_async(dispatch_get_main_queue(), ^{
            if (callback) {
                callback(granted);
            }
        });
    }];
}

QString permissionStatusDescription(PermissionStatus status)
{
    switch (status) {
        case PermissionStatus::Authorized:
            return QStringLiteral("Authorized");
        case PermissionStatus::Denied:
            return QStringLiteral("Denied");
        case PermissionStatus::Restricted:
            return QStringLiteral("Restricted");
        case PermissionStatus::NotDetermined:
            return QStringLiteral("Not Determined");
        default:
            return QStringLiteral("Unknown");
    }
}

} // namespace MacOSAudioPermissions

#endif // Q_OS_MACOS
