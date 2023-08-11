# snowpark-container-demo

CREATE IMAGE REPOSITORY IF NOT EXISTS tm; --same RBAC as your data

CREATE COMPUTE POOL snowpark_container_services
MIN_NODES = 1 --we will guarantee you this
MAX_NODES = 1 --we will take you up to this max
INSTANCE_FAMILY = STANDARD_2 --standard
AUTO_RESUME = TRUE
AUTO_SUSPEND_SECS = 120;

SHOW COMPUTE POOLS like 'SNOWPARK_CONTAINER_SERVICES';
show compute pools;

CREATE SERVICE demo_notebook
MIN_INSTANCES = 1 --replicas
MAX_INSTANCES = 1
COMPUTE_POOL = snowpark_container_services
SPEC = '@snowpark_container_demo/notebook/notebook_manifest.yaml';

CALL SYSTEM$GET_SERVICE_STATUS('SPCS_DEMO.PROD.DEMO_NOTEBOOK');

SHOW SERVICES like 'DEMO_NOTEBOOK';
SELECT "public_endpoints" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

-- Job

CREATE COMPUTE POOL nvidia_a10g
MIN_NODES = 1 --we will guarantee you this
MAX_NODES = 1 --we will take you up to this max
INSTANCE_FAMILY = GPU_5
AUTO_RESUME = TRUE;


--Services Function

ALTER COMPUTE POOL NVIDIA_A10g resume;
SHOW COMPUTE POOLS LIKE 'NVIDIA_A10G';

CREATE SERVICE snowpark_triton
    MIN_INSTANCES = 1
    MAX_INSTANCES = 1
    COMPUTE_POOL = NVIDIA_A10G
    SPEC = @snowpark_container_demo/triton/triton_manifest.yaml;

CALL SYSTEM$GET_SERVICE_STATUS('SPCS_DEMO.PROD.SNOWPARK_TRITON');

SHOW SERVICES like 'SNOWPARK_TRITON';
SELECT "public_endpoints" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));


CREATE OR REPLACE FUNCTION triton(n VARCHAR)
  RETURNS ARRAY
  SERVICE = snowpark_triton
  ENDPOINT = 'tritonclient' --100% secure
  MAX_BATCH_ROWS = 1000
  AS '/inference_snowpark_triton';

SELECT
    review,
    TRITON(review) as embeddings
FROM
    "MUSIC_STORE_REVIEWS"
LIMIT 5000;
