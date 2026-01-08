import unittest
import os
import sys

# Add the current directory to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestApp(unittest.TestCase):
    
    def test_app_import(self):
        """Test that the app can be imported without errors"""
        try:
            import app
            self.assertTrue(True, "App imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import app: {e}")
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        self.assertTrue(os.path.exists('requirements.txt'), "requirements.txt should exist")
    
    def test_readme_exists(self):
        """Test that README.md exists"""
        self.assertTrue(os.path.exists('README.md'), "README.md should exist")
    
    def test_templates_directory_exists(self):
        """Test that templates directory exists"""
        self.assertTrue(os.path.exists('templates'), "templates directory should exist")
    
    def test_index_html_exists(self):
        """Test that index.html exists in templates"""
        self.assertTrue(os.path.exists('templates/index.html'), "templates/index.html should exist")

if __name__ == '__main__':
    unittest.main() 