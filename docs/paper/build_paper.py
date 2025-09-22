#!/usr/bin/env python3
"""
LaTeX Build Script for HumanEvalComm V2 IEEE Paper

This script automates the compilation of the LaTeX document using pdflatex.
It handles multiple compilation passes required for proper bibliography and
cross-reference resolution.

Usage:
    python build_paper.py

Requirements:
    - pdflatex (TeX Live or MiKTeX)
    - Python 3.6+
"""

import sys
import subprocess
import shutil
import platform
from pathlib import Path


class LatexBuilder:
    """LaTeX document builder with automatic compilation management."""

    def __init__(self, source_file="main.tex", output_dir="build"):
        """
        Initialize the LaTeX builder.

        Args:
            source_file (str): Name of the main LaTeX file
            output_dir (str): Directory to store build artifacts
        """
        self.source_file = source_file
        self.output_dir = Path(output_dir)
        self.project_root = Path(__file__).parent
        self.source_path = self.project_root / source_file

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def get_platform(self):
        """Get the current platform for installation commands."""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        elif system == "linux":
            return "linux"
        elif system == "windows":
            return "windows"
        else:
            return "unknown"

    def install_latex_tools(self):
        """
        Attempt to install LaTeX tools automatically.

        Returns:
            bool: True if installation successful, False otherwise
        """
        platform_name = self.get_platform()

        print("🔧 Attempting automatic installation of LaTeX tools...")
        print(f"Detected platform: {platform_name}")
        print()

        if platform_name == "macos":
            return self._install_macos()
        elif platform_name == "linux":
            return self._install_linux()
        elif platform_name == "windows":
            return self._install_windows()
        else:
            print("❌ Automatic installation not supported for this platform")
            print("Please install LaTeX manually:")
            print("  - macOS: brew install mactex")
            print("  - Ubuntu: sudo apt-get install texlive-full")
            print("  - Windows: Install MiKTeX from https://miktex.org/")
            return False

    def _install_macos(self):
        """Install LaTeX on macOS using Homebrew."""
        if not shutil.which("brew"):
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("  /bin/bash -c \"$(curl -fsSL")
            print("  https://raw.githubusercontent.com/Homebrew/install/HEAD/"
                  "install.sh)\"")
            return False

        print("📦 Installing MacTeX via Homebrew...")
        print("⚠️  This may take several minutes and requires ~5GB")
        print("    of disk space")
        print()

        try:
            # Update Homebrew first
            subprocess.run(["brew", "update"], check=True, capture_output=True)

            # Install MacTeX
            subprocess.run(["brew", "install", "mactex"], check=True)

            print("✅ MacTeX installed successfully!")
            print("🔄 You may need to restart your terminal or run:")
            print("  eval \"$(/usr/libexec/path_helper)\"")
            print()

            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False

    def _install_linux(self):
        """Install LaTeX on Linux."""
        # Try to detect the package manager
        package_managers = [
            ("apt-get", ["sudo", "apt-get", "update", "&&",
                         "sudo", "apt-get", "install", "-y", "texlive-full"]),
            ("yum", ["sudo", "yum", "install", "-y", "texlive"]),
            ("dnf", ["sudo", "dnf", "install", "-y", "texlive"]),
            ("pacman", ["sudo", "pacman", "-S", "texlive-most",
                        "texlive-lang"]),
        ]

        for pm_name, install_cmd in package_managers:
            if shutil.which(pm_name):
                print(f"📦 Installing TeX Live via {pm_name}...")
                print("⚠️  This may take several minutes")
                print()

                try:
                    # Run the installation command
                    full_cmd = ["bash", "-c", " ".join(install_cmd)]
                    subprocess.run(full_cmd, check=True)
                    print("✅ TeX Live installed successfully!")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"❌ Installation failed with {pm_name}: {e}")
                    continue

        print("❌ No supported package manager found")
        print("Please install TeX Live manually:")
        print("  Ubuntu/Debian: sudo apt-get install texlive-full")
        print("  CentOS/RHEL: sudo yum install texlive")
        print("  Fedora: sudo dnf install texlive")
        print("  Arch: sudo pacman -S texlive-most texlive-lang")
        return False

    def _install_windows(self):
        """Install LaTeX on Windows."""
        print("📦 Installing MiKTeX on Windows...")
        print("⚠️  This requires administrator privileges")
        print()

        # Try to install MiKTeX using winget or chocolatey
        installers = [
            (["winget", "install", "--id", "MiKTeX.MiKTeX",
              "--accept-source-agreements"], "winget"),
            (["choco", "install", "miktex", "-y"], "Chocolatey"),
        ]

        for install_cmd, installer_name in installers:
            if shutil.which(install_cmd[0]):
                try:
                    subprocess.run(install_cmd, check=True)
                    print("✅ MiKTeX installed successfully!")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"❌ Installation failed with {installer_name}: {e}")
                    continue

        print("❌ No supported installer found")
        print("Please install MiKTeX manually from: https://miktex.org/")
        return False

    def check_requirements(self):
        """Check if required tools are available."""
        required_tools = ['pdflatex', 'bibtex']
        missing_tools = []

        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)

        if missing_tools:
            print(f"❌ Missing required tools: {', '.join(missing_tools)}")
            print()

            try:
                response = input("🤖 Would you like to install LaTeX tools "
                                 "automatically? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    if self.install_latex_tools():
                        print("🔄 Verifying installation...")
                        # Re-check requirements after installation
                        return self.check_requirements()
                    else:
                        print("❌ Automatic installation failed")
                        return False
                else:
                    print("Manual installation instructions:")
            except (EOFError, KeyboardInterrupt):
                print("\nManual installation instructions:")

            # Show manual installation instructions
            platform_name = self.get_platform()
            print("\nPlease install TeX Live or MiKTeX:")
            if platform_name == "macos":
                print("  brew install mactex")
            elif platform_name == "linux":
                print("  Ubuntu/Debian: sudo apt-get install texlive-full")
                print("  CentOS/RHEL: sudo yum install texlive")
                print("  Fedora: sudo dnf install texlive")
                print("  Arch: sudo pacman -S texlive-most texlive-lang")
            elif platform_name == "windows":
                print("  winget install --id MiKTeX.MiKTeX "
                      "--accept-source-agreements")
                print("  Or download from: https://miktex.org/")
            else:
                print("  - macOS: brew install mactex")
                print("  - Ubuntu: sudo apt-get install texlive-full")
                print("  - Windows: Install MiKTeX from https://miktex.org/")
            return False

        print("✅ All required tools found")
        return True

    def run_command(self, command, description):
        """
        Run a shell command and handle errors.

        Args:
            command (list): Command to execute
            description (str): Description for logging

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"🔄 {description}...")
            subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ {description} completed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed:")
            print(f"Error: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return False

    def clean_build_artifacts(self):
        """Clean up intermediate build files."""
        extensions_to_clean = [
            '.aux', '.bbl', '.blg', '.log', '.out', '.toc',
            '.lof', '.lot', '.fdb_latexmk', '.fls'
        ]

        cleaned_files = []
        for ext in extensions_to_clean:
            for file_path in self.project_root.glob(f"*{ext}"):
                file_path.unlink()
                cleaned_files.append(file_path.name)

        if cleaned_files:
            file_list = ', '.join(cleaned_files[:5])
            if len(cleaned_files) > 5:
                file_list += '...'
            print(f"🧹 Cleaned {len(cleaned_files)} build artifacts: "
                  f"{file_list}")

    def build_document(self):
        """
        Build the LaTeX document with proper bibliography processing.

        Returns:
            bool: True if build successful, False otherwise
        """
        base_name = self.source_path.stem

        # Step 1: First pdflatex compilation
        if not self.run_command(
            ['pdflatex',
             '-output-directory', str(self.output_dir),
             self.source_file],
            "First pdflatex compilation"
        ):
            return False

        # Step 2: BibTeX processing
        if not self.run_command(
            ['bibtex', str(self.output_dir / base_name)],
            "BibTeX bibliography processing"
        ):
            return False

        # Step 3: Second pdflatex compilation (resolve citations)
        if not self.run_command(
            ['pdflatex',
             '-output-directory', str(self.output_dir),
             self.source_file],
            "Second pdflatex compilation (citations)"
        ):
            return False

        # Step 4: Third pdflatex compilation (resolve cross-references)
        if not self.run_command(
            ['pdflatex',
             '-output-directory', str(self.output_dir),
             self.source_file],
            "Third pdflatex compilation (cross-references)"
        ):
            return False

        return True

    def build(self):
        """
        Main build method that orchestrates the entire build process.

        Returns:
            bool: True if build successful, False otherwise
        """
        print("🚀 Starting LaTeX document build...")
        print(f"Source: {self.source_file}")
        print(f"Output: {self.output_dir}")
        print()

        # Check requirements
        if not self.check_requirements():
            return False

        # Verify source file exists
        if not self.source_path.exists():
            print(f"❌ Source file not found: {self.source_file}")
            return False

        print(f"📄 Building document: {self.source_file}")
        print()

        # Build document
        if self.build_document():
            pdf_path = self.output_dir / f"{self.source_path.stem}.pdf"
            if pdf_path.exists():
                file_size = pdf_path.stat().st_size / 1024  # KB
                print()
                print("🎉 Build completed successfully!")
                print(f"📋 PDF generated: {pdf_path}")
                print(f"📏 File size: {file_size:.1f} KB")
                print()
                print("📖 The PDF is ready for submission or review.")
                return True
            else:
                print("❌ PDF file was not generated")
                return False
        else:
            print("❌ Build failed")
            return False


def main():
    """Main entry point."""
    builder = LatexBuilder()

    try:
        success = builder.build()

        # Clean up artifacts on success
        if success:
            builder.clean_build_artifacts()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Build interrupted by user")
        return 1
    except (OSError, ValueError, RuntimeError) as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
