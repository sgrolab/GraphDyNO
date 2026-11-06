import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph_dyno.encoders.base import ShapeEncoder
from dino.feature_extractor import ModelType, create_feature_extractor
from dino.dataloader import CUDAPrefetcher
from graph_dyno.precompute.precompute_dino_embeddings import _compute_sdfs_multiprocess, _SimMaskDataset


class DINOEncoder(ShapeEncoder):
    """Shape encoder using a frozen vision transformer (DINOv3 or Cell-DINO).

    Converts binary masks to SDF images, then extracts CLS tokens
    (and optionally mean-pooled patch tokens) via the chosen backbone.
    """

    def __init__(self, model_type: ModelType | str, use_patches: bool = False,
                 sdf_workers: int = 32, target_size: int = 224,
                 backend: str = "hf", repo_dir=None, weights_path=None,
                 weights_url=None, trust_weights: bool = False):
        super().__init__()
        self.model_type = model_type
        self.use_patches = use_patches
        self.sdf_workers = sdf_workers
        self.target_size = target_size
        self.backend = backend

        self.feature_extractor = create_feature_extractor(
            model_type, backend=backend, repo_dir=repo_dir,
            weights_path=weights_path, weights_url=weights_url,
            trust_weights=trust_weights,
        )

        # Resolve model key string for companion file naming
        if isinstance(model_type, ModelType):
            self._model_key = model_type.name
        else:
            self._model_key = model_type

    @property
    def embedding_dim(self) -> int:
        d = self.feature_extractor.embed_dim
        return 2 * d if self.use_patches else d

    @property
    def companion_suffix(self) -> str:
        return f"dino_{self._model_key}"

    @torch.no_grad()
    def encode_masks(
        self,
        masks: torch.Tensor,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode binary masks to DINO embeddings.

        Args:
            masks: (M, H, W) binary masks
            batch_size: Batch size for DINO inference
            num_workers: DataLoader workers

        Returns:
            embeddings: (M, embedding_dim) float32
            crop_sides: (M,) int32
        """
        device = next(self.feature_extractor.buffers()).device

        # Convert binary masks to SDFs via multiprocessing
        sdf_masks, crop_sides = _compute_sdfs_multiprocess(
            masks.cpu(), target_size=self.target_size, num_workers=self.sdf_workers,
        )

        # Batched DINO inference
        ds = _SimMaskDataset(sdf_masks, self.feature_extractor.input_spec)
        loader = DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=False,
        )
        prefetcher = CUDAPrefetcher(loader, device)

        cls_list, pm_list = [], []
        for (imgs,) in tqdm(prefetcher, desc="  DINO inference", total=len(loader), leave=False):
            cls_token, patch_tokens = self.feature_extractor(imgs)
            cls_list.append(cls_token.cpu())
            if self.use_patches:
                pm_list.append(patch_tokens.mean(dim=1).cpu())

        cls_all = torch.cat(cls_list, dim=0)  # (M, D)

        if self.use_patches:
            pm_all = torch.cat(pm_list, dim=0)  # (M, D)
            embeddings = torch.cat([cls_all, pm_all], dim=-1)  # (M, 2D)
        else:
            embeddings = cls_all

        return embeddings, crop_sides
