from args import get_args
from middleware.start_distributed_mode import init_distributed_mode

def test_train(args):

    init_distributed_mode(args=args)



if __name__ == "__main__":
    args = get_args()
    out = test_train(args=args)
    
