import os
import re
from pathlib import Path
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup, Tag
import mlrun

ARTICLE_TOKEN = "Article: "
HEADER_TOKEN = "Subject: "
URLS = ['https://www.iguazio.com/blog/iguazio-releases-data-science-platform-version-2-8/',
        'https://www.iguazio.com/blog/intelligent-edge-iguazio-google/',
        'https://www.iguazio.com/blog/top-9-odsc-europe-sessions-you-cant-miss/',
        'https://www.iguazio.com/blog/cloud-native-will-shake-up-enterprise-storage/',
        'https://www.iguazio.com/blog/building-an-automated-ml-pipeline-with-a-feature-store-using-iguazio-snowflake/',
        'https://www.iguazio.com/blog/concept-drift-and-the-impact-of-covid-19-on-data-science/',
        'https://www.iguazio.com/blog/odsc-east-boston-2022-top-11-sessions-for-ai-and-ml-professionals-to-attend/',
        'https://www.iguazio.com/blog/idc-mlopmarketscape-2022/',
        'https://www.iguazio.com/blog/iguazio-listed-in-7-gartner-hype-cycles-for-2021/',
        'https://www.iguazio.com/blog/announcing-the-winners-mlops-for-good-hackathon/',
        'https://www.iguazio.com/blog/the-importance-of-data-storytelling-in-shaping-a-data-science-product/',
        'https://www.iguazio.com/blog/modernize-it-infrastructure/',
        'https://www.iguazio.com/blog/implementing-automation-and-an-mlops-framework-for-enterprise-scale-ml/',
        'https://www.iguazio.com/blog/automating-ml-pipelines-on-azure-and-azure-stack/',
        'https://www.iguazio.com/blog/real-time-streaming-for-data-science/',
        'https://www.iguazio.com/blog/dcos-apps/',
        'https://www.iguazio.com/blog/iguazio-receives-an-honorable-mention-in-the-2021-gartner-magic-quadrant-for-data-science-and-machine-learning-platforms/',
        'https://www.iguazio.com/blog/gartner-2022-market-guide-for-dsml-engineering-platforms/',
        'https://www.iguazio.com/blog/can-open-source-serverless-be-simpler-than-lambda/',
        'https://www.iguazio.com/blog/cncf-webinar-serverless-ai/',
        'https://www.iguazio.com/blog/2018-can-cloud-big-data-ai-stand-turmoil/',
        'https://www.iguazio.com/blog/2022-predictions/',
        'https://www.iguazio.com/blog/mlops-for-python/',
        'https://www.iguazio.com/blog/mlops-predictions-for-2023/',
        'https://www.iguazio.com/blog/adopting-a-production-first-approach-to-enterprise-ai/',
        'https://www.iguazio.com/blog/from-automl-to-automlops/',
        'https://www.iguazio.com/blog/odscwest2021/',
        'https://www.iguazio.com/blog/top-10-recommended-mlops-world-2021-sessions/',
        'https://www.iguazio.com/blog/breaking-the-silos-between-data-scientists-engineers-and-devops-with-new-mlops-practices/',
        'https://www.iguazio.com/blog/top-8-machine-learning-resources-for-data-scientists-data-engineers-and-everyone/',
        'https://www.iguazio.com/blog/azure-synapse-analytics-and-iguazio/',
        'https://www.iguazio.com/blog/how-to-tap-into-higher-level-abstraction-efficiency-automation-to-simplify-your-ai-ml-journey/',
        'https://www.iguazio.com/blog/how-seagate-runs-advanced-manufacturing-at-scale-with-iguazio/',
        'https://www.iguazio.com/blog/predictive-real-time-operational-ml-pipeline-fighting-customer-churn/',
        'https://www.iguazio.com/blog/build-an-ai-app-in-under-20-minutes/',
        'https://www.iguazio.com/blog/deploying-machine-learning-models-for-real-time-predictions-checklist/',
        'https://www.iguazio.com/blog/data-science-post-hadoop/',
        'https://www.iguazio.com/blog/wanted-a-faster-storage-stack/',
        'https://www.iguazio.com/blog/kubernetes-the-open-scalable-approach-to-ml-pipelines/',
        'https://www.iguazio.com/blog/vmware-on-aws-a-scorecard-for-winners-and-losers/',
        'https://www.iguazio.com/blog/aws-reinvent-data-serverless-ai/',
        'https://www.iguazio.com/blog/beyond-hyped-iguazio-named-in-8-gartner-hype-cycles-for-2022/',
        'https://www.iguazio.com/blog/ai-ml-and-roi-why-your-balance-sheet-cares-about-your-technology-choices/',
        'https://www.iguazio.com/blog/orchestrating-ml-pipelines-scale-kubeflow/',
        'https://www.iguazio.com/blog/using-automated-model-management-for-cpg-trade-success/',
        'https://www.iguazio.com/blog/spark-over-kubernetes/',
        'https://www.iguazio.com/blog/announcing-iguazio-version-3-0-breaking-the-silos-for-faster-deployment/',
        'https://www.iguazio.com/blog/the-complete-guide-to-using-the-iguazio-feature-store-with-azure-ml-part-4/',
        'https://www.iguazio.com/blog/accelerating-ml-deployment-in-hybrid-environments/',
        'https://www.iguazio.com/blog/it-worked-fine-in-jupyter-now-what/',
        'https://www.iguazio.com/blog/kubeflow-vs-mlflow-vs-mlrun/',
        'https://www.iguazio.com/blog/part-one-the-complete-guide-to-using-the-iguazio-feature-store-with-azure-ml/',
        'https://www.iguazio.com/blog/handling-large-datasets-with-mlops-dask-on-kubernetes/',
        'https://www.iguazio.com/blog/faster-ai-development-serverless/',
        'https://www.iguazio.com/blog/nuclio-future-serverless-computing/',
        'https://www.iguazio.com/blog/how-to-build-real-time-feature-engineering-with-a-feature-store/',
        'https://www.iguazio.com/blog/nyc-meetup-jan2018/',
        'https://www.iguazio.com/blog/distributed-feature-store-ingestion-with-iguazio-snowflake-and-spark/',
        'https://www.iguazio.com/blog/iguazio-raises-33m-accelerate-digital-transformation/',
        'https://www.iguazio.com/blog/the-complete-guide-to-using-the-iguazio-feature-store-with-azure-ml-part-2/',
        'https://www.iguazio.com/blog/serverless-can-it-simplify-data-science-projects/',
        'https://www.iguazio.com/blog/machine-learning-hard/',
        'https://www.iguazio.com/blog/free-manufacturing-datasets/',
        'https://www.iguazio.com/blog/building-real-time-ml-pipelines-with-a-feature-store/',
        'https://www.iguazio.com/blog/paving-the-data-science-dirt-road/',
        'https://www.iguazio.com/blog/horovod-for-deep-learning-on-a-gpu-cluster/',
        'https://www.iguazio.com/blog/using-containers-as-mini-vms-is-not-cloud-native/',
        'https://www.iguazio.com/blog/top-9-recommended-odsc-europe-2021-sessions/',
        'https://www.iguazio.com/blog/realtime-bigdata/',
        'https://www.iguazio.com/blog/python-pandas-performance/',
        'https://www.iguazio.com/blog/iguazio-rvmworld-2017-vmware-feeds-off-openstack-decay/',
        'https://www.iguazio.com/blog/how-gpuaas-on-kubeflow-can-boost-your-productivity/',
        'https://www.iguazio.com/blog/mlops-nyc-sessions/',
        'https://www.iguazio.com/blog/2017-predictions-clouds-thunder-and-fog/',
        'https://www.iguazio.com/blog/odsc-east-2023/',
        'https://www.iguazio.com/blog/join-us-at-nvidia-gtc-2021/',
        'https://www.iguazio.com/blog/mckinsey-acquires-iguazio-our-startups-journey/',
        'https://www.iguazio.com/blog/git-based-ci-cd-for-machine-learning-mlops/',
        'https://www.iguazio.com/blog/mlops-for-good-hackathon-roundup/',
        'https://www.iguazio.com/blog/big-data-must-begin-with-clean-slate/',
        'https://www.iguazio.com/blog/suse-iguazio/',
        'https://www.iguazio.com/blog/how-to-run-workloads-on-spark-operator-with-dynamic-allocation-using-mlrun/',
        'https://www.iguazio.com/blog/will-kubernetes-sink-the-hadoop-ship/',
        'https://www.iguazio.com/blog/5-incredible-data-science-solutions-for-real-world-problems/',
        'https://www.iguazio.com/blog/mlops-challenges-solutions-future-trends/',
        'https://www.iguazio.com/blog/cloud-data-services-sprawl-its-complicated/',
        'https://www.iguazio.com/blog/predicting-1st-day-churn-in-real-time/',
        'https://www.iguazio.com/blog/machine-learning-experiment-tracking-from-zero-to-hero-in-2-lines-of-code/',
        'https://www.iguazio.com/blog/how-to-bring-breakthrough-performance-and-productivity-to-ai-ml-projects/',
        'https://www.iguazio.com/blog/how-to-deploy-an-mlrun-project-in-a-ci-cd-process-with-jenkins-pipeline/',
        'https://www.iguazio.com/blog/iguazio-named-in-forresters-now-tech-ai-ml-platforms-q1-2022/',
        'https://www.iguazio.com/blog/the-complete-guide-to-using-the-iguazio-feature-store-with-azure-ml-part-3/',
        'https://www.iguazio.com/blog/what-are-feature-stores-and-why-are-they-critical-for-scaling-data-science/',
        'https://www.iguazio.com/blog/reinventing-data-services/',
        'https://www.iguazio.com/blog/re-structure-in-big-data/',
        'https://www.iguazio.com/blog/top-22-free-healthcare-datasets-for-machine-learning/',
        'https://www.iguazio.com/blog/operationalizing-machine-learning-for-the-automotive-future/',
        'https://www.iguazio.com/blog/automating-mlops-for-deep-learning-how-to-operationalize-dl-with-minimal-effort/',
        'https://www.iguazio.com/blog/iguazio-named-a-fast-moving-leader-by-gigaom-in-the-radar-for-mlops-report/',
        'https://www.iguazio.com/blog/data-science-salon-review-elevating-data-science-practices-for-media-entertainment-advertising/',
        'https://www.iguazio.com/blog/wrapping-up-serverless-nyc-2018/',
        'https://www.iguazio.com/blog/the-next-gen-digital-transformation-cloud-native-data-platforms/',
        'https://www.iguazio.com/blog/best-practices-for-succeeding-with-mlops/',
        'https://www.iguazio.com/blog/did-amazon-just-kill-open-source/',
        'https://www.iguazio.com/blog/cloud-native-storage-primer/',
        'https://www.iguazio.com/blog/serverless-background-challenges-and-future/',
        'https://www.iguazio.com/blog/experiment-tracking/',
        'https://www.iguazio.com/blog/continuous-analytics-real-time-meets-cloud-native/',
        'https://www.iguazio.com/blog/concept-drift-deep-dive-how-to-build-a-drift-aware-ml-system/',
        'https://www.iguazio.com/blog/building-ml-pipelines-over-federated-data-compute-environments/',
        'https://www.iguazio.com/blog/top-8-recommended-mlops-world-2022-sessions/',
        'https://www.iguazio.com/blog/it-vendors-dont-stand-a-chance-against-the-cloud/',
        'https://www.iguazio.com/blog/ml-workflows-what-can-you-automate/',
        'https://www.iguazio.com/blog/iguazio-collaborates-with-equinix-to-offer-data-centric-hybrid-cloud-solutions/',
        'https://www.iguazio.com/blog/gigaom-names-iguazio-a-leader-and-outperformer-for-2022/',
        'https://www.iguazio.com/blog/iguazio-nvidia-edge/',
        'https://www.iguazio.com/blog/extending-kubeflow-into-an-end-to-end-ml-solution/',
        'https://www.iguazio.com/blog/iguazio-listed-in-five-2020-gartner-hype-cycle-reports/',
        'https://www.iguazio.com/blog/data-science-trends-2020/',
        'https://www.iguazio.com/blog/operationalizing-data-science/',
        'https://www.iguazio.com/blog/using-snowflake-and-dask-for-large-scale-ml-workloads/',
        'https://www.iguazio.com/blog/best-13-free-financial-datasets-for-machine-learning/',
        'https://www.iguazio.com/blog/introduction-to-tf-serving/',
        'https://www.iguazio.com/blog/hcis-journey-to-mlops-efficiency/',
        'https://www.iguazio.com/blog/streamlined-iot-at-scale-with-iguazio/',
        'https://www.iguazio.com/blog/iguazio-product-update-optimize-your-ml-workload-costs-with-aws-ec2-spot-instances/',
        'https://www.iguazio.com/blog/top-10-odsc-west-sessions-you-must-attend/',
        'https://www.iguazio.com/blog/iguazio-named-a-leader-and-outperformer-in-gigaom-radar-for-mlops-2022/',
        'https://www.iguazio.com/blog/deploying-your-hugging-face-models-to-production-at-scale-with-mlrun/'
        ]


def normalize(s):
    return s.replace("\n", "").replace("\t", "")


def mark_header_tags(soup):
    nodes = soup.find_all(re.compile("^h[1-6]$"))
    # Tagging headers in html to identify in text files:
    if nodes:
        content_type = type(nodes[0].contents[0])
        nodes[0].string = content_type(
            ARTICLE_TOKEN + normalize(str(nodes[0].contents[0]))
        )
        for node in nodes[1:]:
            if node.string:
                content_type = type(node.contents[0])
                if content_type == Tag:
                    node.string = HEADER_TOKEN + normalize(node.string)
                else:
                    node.string = content_type(HEADER_TOKEN + str(node.contents[0]))


def get_html_as_string(url, mark_headers):
    # read html source:
    req = Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
    web_html_content = urlopen(req).read().decode("utf-8")
    soup = BeautifulSoup(web_html_content, features="html.parser")
    if mark_headers:
        mark_header_tags(soup)
    return soup.get_text()


@mlrun.handler(outputs=["html-as-text-files:directory"])
def collect_html_to_text_files(urls, mark_headers=True):
    directory = "html_as_text_files"
    os.makedirs(directory, exist_ok=True)
    # Writing html files as text files:
    urls = URLS
    for url in urls:
        page_name = Path(url).name
        with open(f"{directory}/{page_name}.txt", "w") as f:
            f.write(get_html_as_string(url, mark_headers))
    return directory