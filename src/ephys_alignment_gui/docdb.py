import json

from aind_data_access_api.document_db import MetadataDbClient
from aind_data_schema_models.modalities import Modality
from aind_data_schema.core.quality_control import QCEvaluation, Stage, QCMetric, QCStatus, Status
from aind_qcportal_schema.metric_value import CurationMetric, CurationHistory
from datetime import datetime
import requests
import boto3
from aws_requests_auth.aws_auth import AWSRequestsAuth

# Resolve DocDB id of data asset
API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

docdb_api_client = MetadataDbClient(
    host=API_GATEWAY_HOST,
    database=DATABASE,
    collection=COLLECTION,
)

def query_docdb_id(session_name: str) -> tuple[str, dict]:
    """
    Returns docdb_id and record for asset_name.
    """

    response = docdb_api_client.retrieve_docdb_records(
        filter_query={"name": {"$regex": f"^{session_name}_sorted*"}}
    )
    if len(response) == 0:
        raise ValueError(f"No ephys sorted record found in docdb for {session_name}")
    
    latest_record = max(response, key=lambda x: x['created']) # pull the most recent ephys sorting record

    docdb_id = latest_record["_id"]
    return docdb_id, latest_record

def write_output_to_docdb(session_name: str, probe: str,
                          channel_results: dict, previous_alignments: dict, ccf_channel_results: dict) -> None:
    """
    writes the output of the IBL gui to docdb. Pulls the latest ephys sorted record and appends qc evaluation

    Parameters
    ----------
    session_name: str
        The name of the session for the probe

    probe: str
        The probe for which the alignment is being done

    channel_results : dict
        Dictionary containing the current channel results from the IBL GUI.
    
    previous_alignments : dict
        Dictionary containing previously stored alignment information, used for comparison or update.

    ccf_channel_results : dict
        Dictionary containing the results aligned to the common coordinate framework (CCF).
    """
    docdb_id = query_docdb_id(session_name)[0]
    # TODO: GET NAME FROM CODEOOCEAN FOR CURATOR
    curation_history = CurationHistory(curator='GET FROM CODEOCEAN', timestamp=datetime.now())
    # use dict
    curations = {'channel_results': channel_results, 'previous_alignments': previous_alignments, 
                 'ccf_channel_results': ccf_channel_results}
    
    curation_metric = CurationMetric(curations=[json.dumps(curations)], curation_history=[curation_history])
    evaluation_name = f'IBL Alignment for {session_name}_{probe}'
    description = 'IBL Probe Alignment of Ephys with Histology'
    qc_metric = QCMetric(name=evaluation_name, description=description, 
                         value=curation_metric, 
                         status_history=[QCStatus(evaluator=curation_history.curator, status=Status.PASS, timestamp=datetime.now())])
    
    evaluation = QCEvaluation(
        modality=Modality.ECEPHYS,
        stage=Stage.PROCESSING,
        name=evaluation_name,
        description=description,
        metrics=[qc_metric]
    )

    session = boto3.Session()
    credentials = session.get_credentials()
    host = "api.allenneuraldynamics.org"

    auth = AWSRequestsAuth(
        aws_access_key=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        aws_token=credentials.token,
        aws_host="api.allenneuraldynamics.org",
        aws_region='us-west-2',
        aws_service='execute-api'
    )
    url = f"https://{host}/v1/add_qc_evaluation"
    post_request_content = {"data_asset_id": docdb_id,
                            "qc_evaluation": evaluation.model_dump(mode='json')}
    response = requests.post(url=url, auth=auth, 
                            json=post_request_content)

    if response.status_code != 200:
        print(response.status_code)
        print(response.text)