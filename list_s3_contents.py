import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the company bucket name from environment variables
COMPANY_BUCKET = os.getenv('COMPANY_S3_BUCKET_NAME')

if not COMPANY_BUCKET:
    raise ValueError("Missing environment variable: COMPANY_S3_BUCKET_NAME")

print(f"Bucket name: {COMPANY_BUCKET}")

# Use the default profile for AWS credentials
session = boto3.Session(profile_name='default')
s3 = session.client('s3')

# List objects in the bucket
print(f"\nListing contents of bucket: {COMPANY_BUCKET}")
print("=" * 50)

try:
    paginator = s3.get_paginator('list_objects_v2')
    count = 0
    
    for page in paginator.paginate(Bucket=COMPANY_BUCKET):
        if 'Contents' in page:
            for obj in page['Contents']:
                count += 1
                size_mb = obj['Size'] / (1024 * 1024)
                print(f"{count}. {obj['Key']} ({size_mb:.2f} MB)")
                
    if count == 0:
        print("Bucket is empty.")
    else:
        print(f"\nTotal objects: {count}")
        
except Exception as e:
    print(f"Error listing bucket contents: {e}")
    print("\nPlease check your AWS credentials and bucket access permissions.") 