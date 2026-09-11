"""Refuses to commit anything that could contain someone's footage.

Run this before every commit. It exists because test output derived from real
frames was once committed and pushed to a public repository: a .gitignore keeps
new files out, but nothing kept me from staging them by hand.

    python tools/staged_check.py
    # exit 0 = nothing suspicious staged
"""

import os
import subprocess
import sys

BANNED_DIRS = ("fisheye_dataset/", "dataset/", "runs/", "tools/struct_out/",
               "tools/ring_out/", "tools/arcade_out/", "tools/dual_out/",
               "tools/detect_out/", "tools/method_out/", "__pycache__/")
ALLOWED_IMAGES = (
    "SteadyFisheye/Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png",
    "icon-1024.png",
)
ALLOWED_SUFFIXES = (".swift", ".metal", ".plist", ".json", ".md", ".py", ".yaml",
                    ".yml", ".pbxproj", ".xcscheme", ".gitignore", ".txt")


def staged_files():
    result = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                            capture_output=True, text=True)
    return [line.strip().replace("\\", "/")
            for line in result.stdout.splitlines() if line.strip()]


def main():
    files = staged_files()
    if not files:
        print("nothing staged")
        return 0

    problems = []
    for path in files:
        if any(path.startswith(d) or f"/{d}" in path for d in BANNED_DIRS):
            problems.append((path, "dataset/test output path"))
            continue
        name = os.path.basename(path)
        if path in ALLOWED_IMAGES:
            continue
        suffix = os.path.splitext(name)[1].lower()
        if suffix in ALLOWED_SUFFIXES or name in (".gitignore", "codemagic.yaml"):
            continue
        if suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".mp4", ".mov", ".ips"):
            problems.append((path, "image or video, could be someone's footage"))
            continue
        problems.append((path, "unexpected file type " + (suffix or "none")))

    print(f"staged: {len(files)} file(s)")
    for path in files:
        print("   ", path)

    if problems:
        print()
        print("REFUSING — review these before committing:")
        for path, reason in problems:
            print(f"    {path}   ({reason})")
        print()
        print("If a file is genuinely intended, add it to ALLOWED_IMAGES in this script.")
        return 1

    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
