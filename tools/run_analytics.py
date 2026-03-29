from pipelines import make_aggregates
from pipelines import exports_bi

def run_analytics():
    make_aggregates.main()
    exports_bi.main()
    return "Exportacao de BI sucesso"
