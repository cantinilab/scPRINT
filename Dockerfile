# Use the specified base image
FROM openproblems/base_pytorch_nvidia:1

# Install Python packages using pip
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
RUN uv pip install --system scprint2

RUN lamin init --storage ./main --name main --modules bionty
RUN lamin load anonymous/main

RUN python -c 'import bionty as bt;bt.core.sync_all_sources_to_latest()'
RUN python -c 'from scdataloader.utils import populate_my_ontology;\
populate_my_ontology(organisms_clade=["vertebrates"],\
sex=["PATO:0000384", "PATO:0000383"],\
organisms=["NCBITaxon:10090", "NCBITaxon:9606"],)'
RUN python -c 'from scdataloader.utils import _adding_scbasecamp_genes; _adding_scbasecamp_genes()'

# Set the default command (can be overridden)
CMD ["scprint2", "--help"]
