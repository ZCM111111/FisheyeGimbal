import SwiftUI
import UIKit

/// Hosts the system share sheet.
///
/// Saving a finished recording into the photo library through the share sheet
/// is deliberate: the save is then performed by iOS itself, with its own UI and
/// its own permission handling. Driving `PHAssetCreationRequest` directly from
/// the app put a fragile call on the automatic path, where a failure takes the
/// whole app down instead of just failing the save.
struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
