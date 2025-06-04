# video-annotation-tool

# 🛠️ Getting Started

Follow the steps below to set up and run the tool on your local computer.

## ✅ Step 1: Install Python

If you don't already have Python installed:
- Go to: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Download and install the latest version for your operating system.
- During installation, **make sure to check the box that says "Add Python to PATH"**.

## ✅ Step 2: Download this Repository

1. Go to [https://github.com/nikhilkuppa/video-annotation-tool](https://github.com/nikhilkuppa/video-annotation-tool)
2. Click the green **Code** button, then choose **Download ZIP**.
3. Extract the ZIP file to a folder you can easily find (e.g., your Desktop).

## ✅ Step 3: Set Up and Run the App

1. Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux).
2. Navigate to the project folder (replace with your actual path):
   ```bash
   cd "C:\Users\YourName\Desktop\video-annotation-tool"
   ```

3. Create a virtual environment:
   ```bash
   python -m venv verificationEnv
   ```

4. Activate the environment:
   
   **Windows:**
   ```bash
   .\verificationEnv\Scripts\activate
   ```
   
   **Mac/Linux:**
   ```bash
   source verificationEnv/bin/activate
   ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

6. Move into the app directory:
   ```bash
   cd app
   ```

7. Start the app:
   ```bash
   python app.py
   ```

## ✅ Step 4: Use the App

After starting the app, you should see something like:
```
Running on http://127.0.0.1:5000
```

Open that link in your web browser.

You can now begin annotating!

## 🙋‍♀️ Need Help?

If you get stuck or run into errors:
- Double check the paths and steps.
- Make sure Python is installed and accessible from the terminal.
