# Use official Node.js image
FROM node:20-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the project
COPY . .

# Expose the server port
EXPOSE 3000

# Start the server
CMD ["npm", "start"]
