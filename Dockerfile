# Use the specified base image
FROM openproblems/base_pytorch_nvidia:1

# Install Python packages using pip
RUN pip install scprint2==0.9.2

RUN lamin init --storage ./main --name main --schema bionty
RUN lamin load anonymous/main

RUN python -c 'import bionty as bt; bt.base.reset_sources(confirm=True);\
bt.core.sync_all_sources_to_latest()'
RUN python -c 'from scdataloader.utils import populate_my_ontology;\
populate_my_ontology(organisms_clade=["vertebrates"],\
sex=["PATO:0000384", "PATO:0000383"],\
organisms=["NCBITaxon:10090", "NCBITaxon:9606"],)'
RUN python -c 'from scprint2.utils import add_scbasecamp_metadata; add_scbasecamp_metadata()'

# Set the default command (can be overridden)
CMD ["scprint2", "--help"]
