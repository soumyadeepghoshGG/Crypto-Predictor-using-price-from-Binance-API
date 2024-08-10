# Future Plan

1. PROPHET BOOST
3. Ensemble Time series model (stacking)
4. GlutonTS with Reticulate
5. Neural Prophet or look into LSTM


 

# To keep your API keys secure by using environment variables, follow these steps:

### **Store API Keys in Environment Variables:**

- **Linux/MacOS**: 
    - Open your terminal and run the following commands to set the environment variables temporarily:
    ```bash
    export BINANCE_API_KEY='your_api_key_here'
    export BINANCE_API_SECRET='your_api_secret_here'
    ```
    - For permanent storage, add these lines to your `~/.bashrc`, `~/.bash_profile`, or `~/.zshrc` file:
    ```bash
    export BINANCE_API_KEY='your_api_key_here'
    export BINANCE_API_SECRET='your_api_secret_here'
    ```

- **Windows**:
    - Open Command Prompt and run:
    ```cmd
    setx BINANCE_API_KEY "your_api_key_here"
    setx BINANCE_API_SECRET "your_api_secret_here"
    ```
    - For temporary usage, you can use:
    ```cmd
    set BINANCE_API_KEY="your_api_key_here"
    set BINANCE_API_SECRET="your_api_secret_here"
    ```