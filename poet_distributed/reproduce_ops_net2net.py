import logging
import numpy as np
from poet_distributed.niches.box2d.model_net2net import ModelNet2Net
from poet_distributed.niches.box2d.env_net2net import bipedhard_custom

logger = logging.getLogger(__name__)


class ReproducerNet2Net:
    def __init__(self, args):
        self.rs = np.random.RandomState(args.master_seed)

    def mutate(self, theta):
        model = ModelNet2Net(bipedhard_custom)
        model.set_model_params(theta)

        specs_length = len(model.weight) - 1
        widen_specs = self.rs.randint(low=0, high=8, size=specs_length)
        deepen_specs = [0] * specs_length
        if specs_length < 10:
            deepen_specs[self.rs.randint(low=0, high=specs_length, size=1)[0]] = self.rs.binomial(n=1, p=0.7, size=1)[0]

        logger.info(f"Shape before net2net: {model.shapes}")
        model.net2widernet(widen_specs, quiet_mode=True)
        model.net2deepernet(deepen_specs, quiet_mode=True)
        logger.info(f"Shape after net2net: {model.shapes}")

        return model.get_model_params()
