#!/bin/bash

# Path to the configuration file
CONFIG_FILE="config.yaml"

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Extract from the YAML file
DATA_URL=$(grep '^data_url:' "$CONFIG_FILE" | awk '{print $2}' | sed 's/"//g')
SAVE_PATH=$(grep '^db_path:' "$CONFIG_FILE" | awk '{print $2}' | sed 's/"//g')

# Ensure the extracted paths are valid
if [ -z "$DATA_URL" ] || [ -z "$SAVE_PATH" ]; then
    echo "❌ Error: Failed to extract DATA_URL or SAVE_PATH from '$CONFIG_FILE'."
    exit 1
fi

# Check if the dataset exists
if [ ! -f "$SAVE_PATH" ]; then
    echo "⚠️  Dataset '$SAVE_PATH' not found in the 'data' folder."
    
    # Prompt the user to download the dataset automatically
    read -p "   Do you want to download the dataset automatically? (y/n): " choice
    
    # Convert the input to lowercase for case-insensitive comparison
    choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')
    
    if [ "$choice" == "y" ] || [ "$choice" == "yes" ]; then
        echo "   └── Creating directory for $SAVE_PATH if it doesn't exist..."
        mkdir -p "$(dirname "$SAVE_PATH")"

        echo "   └── Downloading from $DATA_URL..."
        if curl -o "$SAVE_PATH" "$DATA_URL" --fail --silent --show-error; then
            echo "   └── Saving to $SAVE_PATH..."
            echo
            echo "✅ Dataset downloaded successfully!"
            echo
        else
            echo "❌ Error: Download failed."
            exit 1
        fi
    elif [ "$choice" == "n" ] || [ "$choice" == "no" ]; then
        echo "   Exiting pipeline. Please manually place the dataset in the 'data' folder and try again."
        exit 1
    else
        echo "❌ Invalid input. Please run the command again and enter 'y' or 'n'."
        echo "   Exiting pipeline."
        exit 1
    fi
fi

# Parse optional arguments
if [[ "$1" == "--lite" ]]; then
    echo "🚀🚀💡 Running pipeline in in Lite Mode... (<1 min)"
    echo
    python src/pipeline.py --lite
elif [[ $# -gt 0 ]]; then
    # Unknown argument
    echo "❌ Unknown argument: $1"
    exit 1
else
    # Run the full machine learning pipeline with configurable parameters
    echo "🚀🚀🚀 Running the machine learning pipeline... (~5 min)"
    echo
    python src/pipeline.py 
fi

# Check if the pipeline executed successfully
if [ $? -eq 0 ]; then
    echo "🍻 Pipeline executed successfully!"
else
    echo
    echo "❌ Error: Pipeline execution failed."
    exit 1
fi