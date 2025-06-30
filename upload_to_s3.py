import boto3
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
import glob

# Load environment variables
load_dotenv()

# Local directory with files to upload
LOCAL_DIR = '/Users/adamimos/Downloads/results'

# Target S3 bucket name
COMPANY_BUCKET = os.getenv('COMPANY_S3_BUCKET_NAME')

if not COMPANY_BUCKET:
    raise ValueError("Missing environment variable: COMPANY_S3_BUCKET_NAME")

print(f"Target bucket: {COMPANY_BUCKET}")
print(f"Source directory: {LOCAL_DIR}")

# Check if source directory exists
if not os.path.exists(LOCAL_DIR):
    raise ValueError(f"Source directory does not exist: {LOCAL_DIR}")

# Use the default profile for AWS credentials
session = boto3.Session(profile_name='default')
s3 = session.client('s3')

# Verify bucket access
print("\nVerifying bucket access...")
try:
    test_key = "test_permission_delete_me"
    s3.put_object(Bucket=COMPANY_BUCKET, Key=test_key, Body="test")
    print(f"✅ Bucket write access OK")
    s3.delete_object(Bucket=COMPANY_BUCKET, Key=test_key)
    print(f"✅ Bucket delete access OK")
except Exception as e:
    print(f"❌ Cannot write to bucket: {e}")
    print("\nPlease check your AWS credentials and bucket access permissions.")
    raise ValueError("No access to bucket")

# List all files in the source directory and subdirectories
def list_files(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip hidden files like .DS_Store
            if file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

# Get all files
files_to_upload = list_files(LOCAL_DIR)
print(f"\nFound {len(files_to_upload)} files to upload")

# Preview first 5 files
if files_to_upload:
    print("\nFirst 5 files to upload:")
    for i, file in enumerate(files_to_upload[:5]):
        print(f"{i+1}. {file}")

# Preview sample S3 keys for first few files
print("\nSample S3 keys (destination paths):")
for i, file_path in enumerate(files_to_upload[:3]):
    # Create S3 key that matches the existing structure
    rel_path = os.path.relpath(file_path, LOCAL_DIR)
    s3_key = f"quantum_runs/{rel_path}"
    print(f"{i+1}. {file_path} -> {s3_key}")

# Confirm upload
confirm = input("\nDo you want to upload these files with the above structure? (yes/no): ")
if confirm.lower() != 'yes':
    print("Upload cancelled")
    exit()

# Upload files
print("\nUploading files...")
uploaded_count = 0
failed_count = 0
failed_files = []

for file_path in tqdm(files_to_upload, desc="Uploading"):
    # Create the S3 key to match existing structure (quantum_runs/[date]/[run]/[file])
    rel_path = os.path.relpath(file_path, LOCAL_DIR)
    s3_key = f"quantum_runs/{rel_path}"
    
    try:
        # Upload the file
        s3.upload_file(file_path, COMPANY_BUCKET, s3_key)
        uploaded_count += 1
    except Exception as e:
        print(f"\nError uploading {file_path}: {e}")
        failed_count += 1
        failed_files.append(file_path)

# Print summary
print("\nUpload Summary:")
print(f"✅ Successfully uploaded: {uploaded_count} files")
if failed_count > 0:
    print(f"❌ Failed to upload: {failed_count} files")
    print("\nFailed files:")
    for file in failed_files:
        print(f"- {file}")
else:
    print("All files uploaded successfully!") 