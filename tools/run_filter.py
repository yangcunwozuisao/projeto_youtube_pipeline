from pipelines import step_eda
from pipelines import step_filter

def run_filter():
    step_eda.main()
    step_filter.main()
    return "Filtracao finalizada"
