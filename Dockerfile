
# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entry point for the container
ENTRYPOINT ["torchrun", "--nproc_per_node=auto", "train.py"]
