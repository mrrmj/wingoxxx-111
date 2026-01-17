
from cryptography.fernet import Fernet
import os

# Generate a key and keep it safe
# key = Fernet.generate_key()
# print(key)

# For demonstration, use a fixed key or load from environment variable
# In production, load from a secure environment variable or key management service
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", "YOUR_GENERATED_FERNET_KEY").encode()
fernet = Fernet(ENCRYPTION_KEY)

def encrypt_data(data):
    return fernet.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data):
    return fernet.decrypt(encrypted_data.encode()).decode()

if __name__ == "__main__":
    # Example usage
    original_data = "This is sensitive data."
    encrypted = encrypt_data(original_data)
    decrypted = decrypt_data(encrypted)
    print(f"Original: {original_data}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")


