FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Create a non-root user and grant permissions to the workdir
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home appuser && \
    chown -R appuser:appuser /app

# Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install packages
RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install -r /app/requirements.txt --no-cache-dir

# Copy the rest of the application code
COPY --chown=appuser:appuser . /app

# Switch to the non-root user
USER appuser

CMD ["python", "test.py", "--config", "configs/test.yaml"]
