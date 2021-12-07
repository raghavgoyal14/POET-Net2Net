# POET-Net2Net

This repo contains implementation of [Net2Net](https://arxiv.org/abs/1511.05641) within [Original-POET](https://arxiv.org/abs/1901.01753) algorithm described in:

## Integration of Net2Net
- POET's implementation uses `theta` to decribe agent's parameters in flattened format
- To desribe an agent together with its architecture specs completely using `theta`, I appended the specification of layers in the beginning,

  `theta_new = [num_layers, num_units_layer_1, num_units_layer_2, ..., num_units_layer_n-1, theta_old]`

  This helps plugging in modified architectures in POET's code as it uses `theta` to completely specify an agent
- Net2Net wider can be found here: https://github.com/raghavgoyal14/POET-Net2Net/blob/main/poet_distributed/niches/box2d/model_net2net.py#L262
- Net2Net deeper can be found here: https://github.com/raghavgoyal14/POET-Net2Net/blob/main/poet_distributed/niches/box2d/model_net2net.py#L335
- Mutating agent's network in POET algorithm is here: https://github.com/raghavgoyal14/POET-Net2Net/blob/main/poet_distributed/poet_algo_net2net.py#L338
- Mutating specs (probabilities for widen and deeper) can be found here: https://github.com/raghavgoyal14/POET-Net2Net/blob/main/poet_distributed/reproduce_ops_net2net.py




## Requirements

- [Fiber](https://uber.github.io/fiber/)
- [OpenAI Gym](https://github.com/openai/gym)

## Run the code locally

To run locally on a multicore machine

```./run_poet_local.sh final_test```

## Run the code on a computer cluster

To containerize and run the code on a computer cluster (e.g., Google Kubernetes Engine on Google Cloud), please refer to [Fiber Documentation](https://uber.github.io/fiber/getting-started/#containerize-your-program).
