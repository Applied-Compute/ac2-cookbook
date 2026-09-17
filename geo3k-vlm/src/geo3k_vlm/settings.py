import os
from ac2.sdk import Client


def client():
    return Client(project=os.environ["AC2_PROJECT"], active_org_id=os.environ["AC2_ORG_ID"])
