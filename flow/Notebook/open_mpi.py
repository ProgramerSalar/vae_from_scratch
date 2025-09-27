import os 


def open_mpi():
    if int(os.getenv('OMPI_COMM_WORLD_SIZE', '0')) > 0:
        # print('working...')
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    