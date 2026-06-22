# S Voice Server

A custom server for S Voice that makes it work again locally. Works with S Voice versions that shipped until Android 5, tested with 11.2.2.0.x.

<p align="center">
  <img src="https://github.com/user-attachments/assets/870095ea-9a97-4262-a512-8d1a309bc969" width="200" />
  <img src="https://github.com/user-attachments/assets/3eb900c1-e1bf-4c35-bbcc-ac9c68832432" width="200" />
</p>

<details>
  <summary>All tested versions</summary>
  <ul>
    <li>11.2.2.0.39002 (r1) on S4 (I9505)</li>
    <li>11.2.2.0.37791 (r1) on Note 3 (N9005)</li>
    <li>11.2.2.0.32948 (r1) on S3 Mini (I8200)</li>
    <li>11.2.2.0.32564 (r1) on S3 Neo (I9301I)</li>
    <li>11.2.2.0.33209 (r1) on S4 Mini (I9195)</li>
  </ul>
</details>

## How it works

It works by redirecting requests sent from S Voice to a local server. The server then decodes the audio and uses standard speech-to-text libraries to transcribe it and passes it to an if-else chain. The chain then chooses whether to respond with plain text information or an action such as setting a timer, opening an app, etc.

The current supported actions are opening apps, setting timers, toggling bt/wifi, starting voice recordings, searching the web, and controlling music.

## Setup

### Prerequesites

- New-ish Python version (3.13.5 tested)
- mitmproxy (`brew install mitmproxy`)
- ffmpeg in path (`brew install ffmpeg`)
- Wi-Fi on the phone
  
### Configure the server

Open `config.json` and update the settings:

Server Configuration:
- `server.host`: Your local IP address for redirects (find with `ifconfig`, `ip addr`, or `ipconfig`)
- `server.bind`: Address to bind the server to (default: "0.0.0.0" to accept connections from any interface)
- `server.port`: Server port (default 5067)

Options:
- `options.keep_audio_files`: Set to true to keep audio files
- `options.ffmpeg_path`: Path to ffmpeg binary (default: "ffmpeg" - uses system PATH)

### Start the proxy server

This allows the computer to read and modify http traffic on the phone

- The svoiceredir.py script is used along with mitmproxy to achieve that
- Run the `run_svoiceredir.sh` script, or just `mitmproxy -s svoiceredir.py`
- It will run at port 8080 by default
- Allow mitmproxy through your firewall if needed

### Configure the phone

The phone needs to be configured to use our proxy server

- This may vary across devices, but is generally something like this
- Go to Settings > Wi-Fi and forget your current network if already connected
- Tap on your network, input your password but do not connect yet
- Check "Show Advanced Options"
- In the proxy settings, select "Manual"
- In the ip address field, enter your computer's ip address (from config.json)
- In the port field, enter 8080 unless you have changed the port manually during step 2
- If you open S Voice, you should see a failing request in the mitmproxy window from step 2

### Setup the Python environment

Now we need to set up the S Voice server itself

- You need the packages listed in requirements.txt
- To install the packages: `pip install -r requirements.txt` or `uv sync`

> [!NOTE]
> Installing the requirements in a venv instead of the global Python installation is recommended.
> 
> `python -m venv .venv` to create it, `source .venv/bin/activate` to activate on macOS/Linux, `.venv\Scripts\activate` on Windows

### Start the server

- Type `python main.py` to start the server
- If everything is set up correctly, you should see `Application startup complete.`
- Visit http://localhost:5067/dash (or your configured port) to access the dashboard
