FROM node:20-slim

# Install python3 and build tools for node-gyp
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies and tell node-gyp to use python3
RUN npm install --python=/usr/bin/python3

# Copy rest of project
COPY . .

# Expose the server port
EXPOSE 3291

# Start the server
CMD ["npm", "start"]
