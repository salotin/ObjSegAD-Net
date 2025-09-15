import os
import cv2
import torch
from u2net import U2NET  # 确保模型定义和权重兼容
from u2net_utils import load_image, post_process

# 默认配置
DEFAULT_RESIZE = 256
DEFAULT_MODEL_PATH = "models/u2net.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集配置
DATASET_CONFIGS = {
    'btad': {
        'data_root': "./dataset/BTech_Dataset_transformed",
        'input_subdir': os.path.join("train", "ok"),
        'output_subdir': os.path.join("train_mask", "ok"),
        'supported_extensions': (".jpg", ".png", ".bmp"),
        'use_product_ids': True,
        'default_products': ['01', '02', '03']
    },
    'dagm': {
        'data_root': "./data/MVTec",
        'input_subdir': os.path.join("train", "good"),
        'output_subdir': os.path.join("train_mask", "good"),
        'supported_extensions': (".jpg", ".png"),
        'use_product_ids': False
    },
    'mpdd': {
        'data_root': "./dataset/MPDD",
        'input_subdir': os.path.join("train", "good"),
        'output_subdir': os.path.join("train_mask", "good"),
        'supported_extensions': (".jpg", ".png", ".jpeg", ".bmp"),
        'use_product_ids': False
    },
    'mvtec': {
        'data_root': "./data/MVTec",
        'input_subdir': os.path.join("train", "good"),
        'output_subdir': os.path.join("train_mask", "good"),
        'supported_extensions': (".jpg", ".png"),
        'use_product_ids': False
    },
    'visa': {
        'data_root': "/root/autodl-tmp/objseg/dataset",
        'input_subdir': os.path.join("train", "good"),
        'output_subdir': os.path.join("train_mask", "good"),
        'supported_extensions': (".jpg", ".png"),
        'use_product_ids': False
    }
}

class UnifiedMaskGenerator:
    """统一的前景掩码生成器，支持多种异常检测数据集"""
    
    def __init__(self, dataset_type, data_root=None, model_path=None, resize=None, device=None):
        """
        初始化掩码生成器
        
        Args:
            dataset_type (str): 数据集类型 ('btad', 'dagm', 'mpdd', 'mvtec', 'visa')
            data_root (str, optional): 数据根目录，如果为None则使用默认配置
            model_path (str, optional): U2Net模型路径
            resize (int, optional): 图像resize尺寸
            device (torch.device, optional): 计算设备
        """
        if dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Supported types: {list(DATASET_CONFIGS.keys())}")
        
        self.dataset_type = dataset_type
        self.config = DATASET_CONFIGS[dataset_type].copy()
        
        # 覆盖默认配置
        if data_root is not None:
            self.config['data_root'] = data_root
        
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.resize = resize or DEFAULT_RESIZE
        self.device = device or DEVICE
        
        # 初始化模型
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载U2Net模型"""
        self.model = U2NET(3, 1)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device).eval()
        print(f"Model loaded from {self.model_path}")
    
    def process_category(self, category_name):
        """
        处理指定类别的所有图像
        
        Args:
            category_name (str): 类别名称（对于BTAD是产品ID，对于其他数据集是对象名称）
        """
        input_dir = os.path.join(self.config['data_root'], category_name, self.config['input_subdir'])
        output_dir = os.path.join(self.config['data_root'], category_name, self.config['output_subdir'])
        
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory not found: {input_dir}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing {self.dataset_type.upper()} category: {category_name}")
        
        processed_count = 0
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(self.config['supported_extensions'])]
        
        if not image_files:
            print(f"No valid images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for fname in sorted(image_files):
            img_path = os.path.join(input_dir, fname)
            raw = cv2.imread(img_path)
            
            if raw is None:
                print(f"Warning: Failed to load image: {img_path}")
                continue
            
            h, w = raw.shape[:2]
            
            # 生成掩码
            image = load_image(img_path, self.resize).to(self.device)
            with torch.no_grad():
                d1, *_ = self.model(image)
            mask = post_process(d1, h, w)
            
            # 保存掩码，文件名添加_mask后缀
            base, ext = os.path.splitext(fname)
            new_fname = f"{base}_mask{ext}"
            mask_path = os.path.join(output_dir, new_fname)
            cv2.imwrite(mask_path, mask)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(image_files)} images")
        
        print(f"Successfully processed {processed_count} images for {category_name}")
    
    def process_all_categories(self, categories=None):
        """
        处理所有类别
        
        Args:
            categories (list, optional): 要处理的类别列表，如果为None则自动发现所有类别
        """
        if categories is None:
            # 自动发现类别
            if self.dataset_type == 'btad' and self.config.get('use_product_ids'):
                # BTAD使用预定义的产品ID
                categories = self.config.get('default_products', [])
                # 同时检查数据目录中实际存在的产品
                if os.path.exists(self.config['data_root']):
                    existing_products = [d for d in os.listdir(self.config['data_root']) 
                                       if os.path.isdir(os.path.join(self.config['data_root'], d))]
                    categories = [p for p in categories if p in existing_products]
            else:
                # 其他数据集自动发现所有对象类别
                if os.path.exists(self.config['data_root']):
                    categories = [d for d in os.listdir(self.config['data_root']) 
                                if os.path.isdir(os.path.join(self.config['data_root'], d))]
                else:
                    print(f"Data root directory not found: {self.config['data_root']}")
                    return
        
        if not categories:
            print(f"No categories found for {self.dataset_type} dataset")
            return
        
        print(f"Found categories: {categories}")
        
        for category in categories:
            self.process_category(category)
        
        print(f"\nAll {self.dataset_type.upper()} foreground masks generated successfully!")

# 向后兼容的函数
def process_btad_category(product_id, data_root=None, model_path=None):
    """BTAD数据集的向后兼容函数"""
    generator = UnifiedMaskGenerator('btad', data_root, model_path)
    generator.process_category(product_id)

def process_category(object_name, dataset_type='mvtec', data_root=None, model_path=None):
    """通用的向后兼容函数"""
    generator = UnifiedMaskGenerator(dataset_type, data_root, model_path)
    generator.process_category(object_name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate foreground masks for anomaly detection datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=list(DATASET_CONFIGS.keys()),
                       help='Dataset type')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Data root directory (optional, uses default if not specified)')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                       help='Path to U2Net model')
    parser.add_argument('--categories', type=str, nargs='*', default=None,
                       help='Specific categories to process (optional, processes all if not specified)')
    parser.add_argument('--resize', type=int, default=DEFAULT_RESIZE,
                       help='Image resize dimension')
    
    args = parser.parse_args()
    
    # 创建掩码生成器
    generator = UnifiedMaskGenerator(
        dataset_type=args.dataset,
        data_root=args.data_root,
        model_path=args.model_path,
        resize=args.resize
    )
    
    # 处理指定类别或所有类别
    if args.categories:
        for category in args.categories:
            generator.process_category(category)
    else:
        generator.process_all_categories()