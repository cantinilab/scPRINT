# Use the specified base image
FROM openproblems/base_pytorch_nvidia:1

# Install Python packages using pip
RUN pip install git+https://github.com/cantinilab/scPRINT.git@d8cc270b099c8d5dacf6913acc26f2b696685b2b
RUN pip install gseapy==1.1.2
RUN pip install git+https://github.com/jkobject/scDataLoader.git@c67c24a2e5c62399912be39169aae76e29e108aa

RUN lamin init --storage ./main --name main --schema bionty
RUN lamin load anonymous/main

RUN python -c 'import bionty as bt; bt.base.reset_sources(confirm=True); bt.core.sync_all_sources_to_latest()'
RUN python -c 'from scdataloader.utils import populate_my_ontology; populate_my_ontology()'

# Set the default command (can be overridden)
CMD ["scprint", "--help"]
