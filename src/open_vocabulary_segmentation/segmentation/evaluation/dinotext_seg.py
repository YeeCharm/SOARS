import mmcv
import mmengine
import torch
import torch.nn.functional as F
from mmseg.models import EncoderDecoder
from utils import get_logger


class DINOTextSegInference(EncoderDecoder):
    def __init__(
            self,
            model,
            text_embedding,
            classnames,
            with_bg,
            test_cfg=dict(),
            pamr=False,
            bg_thresh=0.5,
            bg_strategy="base",
            # kp_w=0.3,
            **kwargs,
    ):
        # mmseg 1.x EncoderDecoder init requires different args, 
        # but we are hacking it to use our own model.
        # We initialize the base class with minimal args to avoid errors.
        # Note: EncoderDecoder inherits from BaseSegmentor -> BaseModel -> BaseModule
        
        # Bypass EncoderDecoder.__init__ which expects backbone, decode_head etc.
        # We directly init the grand-parent BaseSegmentor (if possible) or just nn.Module
        # But EncoderDecoder has useful methods. Let's try to init it with dummy args if needed,
        # or just call super(EncoderDecoder, self).__init__() if it allows.
        # Actually, EncoderDecoder.__init__ calls super().__init__() and then builds components.
        # If we don't provide backbone config, it might fail.
        # Let's try to just init nn.Module and mixin what we need, or mock the init.
        
        # For now, let's try calling super().__init__() with empty args and hope it doesn't crash
        # or we manually set the attributes it needs.
        # mmseg 1.x EncoderDecoder.__init__(self, backbone, decode_head, neck=None, auxiliary_head=None, train_cfg=None, test_cfg=None, data_preprocessor=None, init_cfg=None)
        
        # Since we are wrapping an existing model, we don't want EncoderDecoder to build anything.
        # So we just call nn.Module.__init__(self) and manually set test_cfg.
        torch.nn.Module.__init__(self)
        
        if not isinstance(test_cfg, mmengine.Config):
            try:
                test_cfg = mmengine.Config(test_cfg)
            except:
                pass # test_cfg might be a dict
        self.test_cfg = test_cfg
        self.pamr = pamr
        self.bg_thresh = bg_thresh
        self.bg_strategy = bg_strategy
        # self.kp_w = kp_w

        self.model = model
        self.register_buffer("text_embedding", text_embedding)
        self.classnames = classnames
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(text_embedding) + 1
        else:
            self.num_classes = len(text_embedding)

        self.align_corners = False
        logger = get_logger()
        logger.info(
            f"Building DINOTextSegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}"
            f", pamr={pamr}, bg_thresh={bg_thresh}"
        )

    def predict(self, inputs, data_samples):
        """
        New method for mmseg 1.x inference.
        Args:
            inputs (Tensor): The input tensor with shape (N, C, H, W).
            data_samples (List[SegDataSample]): The data samples.
        """
        img = inputs
        img_metas = [sample.metainfo for sample in data_samples]
        
        # Call encode_decode
        seg_logits = self.encode_decode(img, img_metas)
        
        return seg_logits

    def _forward(self, inputs, data_samples):
        """
        Placeholder for _forward
        """
        return self.predict(inputs, data_samples)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        """
        # assert img.shape[0] == 1, "batch size must be 1"

        # masks [B, N, H, W]
        # simmap [B, N, H//4, W//4]
        # soft mask (logit-like) is required
        masks, simmap = self.model.generate_masks(
            img,
            img_metas,
            self.text_embedding,
            self.classnames,
            apply_pamr=self.pamr,
            # kp_w=self.kp_w,
        )

        B, N, H, W = masks.shape

        if self.with_bg:

            masks = masks.cpu()

            background = torch.full(
                [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
            )
            masks = torch.cat([background, masks], dim=1)
            masks = masks.to(img.device)

        return masks
