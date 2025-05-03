#!/bin/bash

# Reset the project by deleting data, models, and output folders
echo "🔄 Resetting the project..."

# Define directories to delete
DIRECTORIES=("data" "models" "output")

# Loop through each directory and delete if it exists
for DIR in "${DIRECTORIES[@]}"; do
    if [ -d "$DIR" ]; then
        echo "   └── Deleting '$DIR' folder..."
        rm -rf "$DIR"
    else
        echo "   ⚠️  '$DIR' folder does not exist. Skipping deletion."
    fi
done

# Final confirmation
echo
echo "✅ Project reset completed successfully!"