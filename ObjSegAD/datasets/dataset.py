import os
import random
import zipfile
import requests
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class UnifiedAnomalyDataset(Dataset):
    """
    统一的异常检测数据集加载器，支持多种数据集格式：
    - BTAD: 工业缺陷检测数据集
    - DAGM: DAGM 2007 数据集
    - MPDD: 金属零件缺陷数据集
    - MVTec: MVTec AD 数据集
    - VisA: VisA 数据集
    
    参数:
        root_dir (str): 数据集根目录路径
        dataset_type (str): 数据集类型，可选 'btad', 'dagm', 'mpdd', 'mvtec', 'visa'
        category (str): 数据集类别/产品名称
        mode (str): 'train' 或 'test'
        resize (int): 图像缩放尺寸
        anomaly_source_dir (str, optional): 外部异常纹理目录
    """
    
    def __init__(self, root_dir, dataset_type, category='all', mode='train', 
                 resize=256, anomaly_source_dir=None):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_type = dataset_type.lower()
        self.category = category
        self.mode = mode
        self.resize = resize
        self.anomaly_source_dir = anomaly_source_dir
        
        # 初始化数据路径列表
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        # 根据数据集类型初始化
        if self.dataset_type == 'btad':
            self._init_btad()
        elif self.dataset_type == 'dagm':
            self._init_dagm()
        elif self.dataset_type == 'mpdd':
            self._init_mpdd()
        elif self.dataset_type == 'mvtec':
            self._init_mvtec()
        elif self.dataset_type == 'visa':
            self._init_visa()
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def _init_btad(self):
        """初始化BTAD数据集"""
        # 自动下载逻辑
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)
        subdirs = [d for d in os.listdir(self.root_dir)
                   if os.path.isdir(os.path.join(self.root_dir, d))]
        if not any(d.isdigit() for d in subdirs):
            self._download_btad()
        
        # 定位产品目录
        if self.category == 'all':
            candidates = ['01','02','03','product_1','product_2','product_3']
        else:
            candidates = [self.category.zfill(2), f"product_{self.category}"]
        
        self.prod_root = None
        for d in candidates:
            p = os.path.join(self.root_dir, d)
            if os.path.isdir(p):
                self.prod_root = p
                break
        
        if self.prod_root is None:
            raise RuntimeError(f"BTAD产品文件夹未找到: {candidates} in {self.root_dir}")
        
        # 收集图像路径
        ok_names = ['good', 'ok']
        anom_names = ['anomalous', 'ko']
        
        if self.mode == 'train':
            # 训练集只需正常样本
            for name in ok_names:
                td = os.path.join(self.prod_root, 'train', name)
                if os.path.isdir(td):
                    for fn in sorted(os.listdir(td)):
                        if fn.lower().endswith(('.png','.jpg','.bmp','.jpeg')):
                            self.image_paths.append(os.path.join(td, fn))
                    break
        else:
            # 测试集：正常样本
            for name in ok_names:
                td = os.path.join(self.prod_root, 'test', name)
                if os.path.isdir(td):
                    for fn in sorted(os.listdir(td)):
                        if fn.lower().endswith(('.png','.jpg','.bmp','.jpeg')):
                            self.image_paths.append(os.path.join(td, fn))
                            self.labels.append(0)
                            self.mask_paths.append(None)
                    break
            
            # 测试集：异常样本
            for name in anom_names:
                td = os.path.join(self.prod_root, 'test', name)
                if os.path.isdir(td):
                    for fn in sorted(os.listdir(td)):
                        if fn.lower().endswith(('.png','.jpg','.bmp','.jpeg')):
                            imgp = os.path.join(td, fn)
                            self.image_paths.append(imgp)
                            self.labels.append(1)
                            
                            # 查找对应掩码
                            stem = os.path.splitext(fn)[0]
                            mask_dir = os.path.join(self.prod_root, 'ground_truth', name)
                            found = None
                            for ext in ('.png','.jpg','.bmp','.jpeg'):
                                cand = os.path.join(mask_dir, stem + ext)
                                if os.path.exists(cand):
                                    found = cand
                                    break
                            self.mask_paths.append(found)
                    break
        
        # 收集掩码形状和纹理路径
        self._collect_btad_resources()
    
    def _download_btad(self):
        """下载BTAD数据集"""
        zip_url = "http://avires.dimi.uniud.it/papers/btad/btad.zip"
        zip_path = os.path.join(self.root_dir, "btad.zip")
        try:
            print("正在下载BTAD数据集 (6 GB)...")
            r = requests.get(zip_url, stream=True)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("正在解压BTAD数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zr:
                zr.extractall(self.root_dir)
        except Exception as e:
            print(f"下载失败 ({e})，请手动准备数据到 {self.root_dir}")
    
    def _collect_btad_resources(self):
        """收集BTAD的掩码形状和纹理资源"""
        self.mask_shapes = []
        anom_names = ['anomalous', 'ko']
        gt_base = os.path.join(self.prod_root, 'ground_truth')
        for name in anom_names:
            gt_dir = os.path.join(gt_base, name)
            if os.path.isdir(gt_dir):
                for fn in os.listdir(gt_dir):
                    if fn.lower().endswith(('.png','.jpg','.bmp','.jpeg')):
                        self.mask_shapes.append(os.path.join(gt_dir, fn))
                break
        
        self.texture_paths = []
        if self.anomaly_source_dir and os.path.isdir(self.anomaly_source_dir):
            for root, _, files in os.walk(self.anomaly_source_dir):
                for fn in files:
                    if fn.lower().endswith(('.png','.jpg','.bmp','.jpeg')):
                        self.texture_paths.append(os.path.join(root, fn))
    
    def _init_dagm(self):
        """初始化DAGM数据集"""
        # 枚举要加载的类别目录
        if self.category == 'all':
            classes = [d for d in os.listdir(self.root_dir) if d.startswith('Class')]
        else:
            classes = [f'Class{self.category}']
        
        for cls in classes:
            # 处理双层 ClassX/ClassX 结构
            cls_root = os.path.join(self.root_dir, cls)
            nested = os.path.join(cls_root, cls)
            if os.path.isdir(nested):
                cls_root = nested
            
            # 训练/测试主目录
            train_dir = os.path.join(cls_root, 'Train')
            test_dir = os.path.join(cls_root, 'Test')
            
            if self.mode == 'train':
                # Train 下所有 .PNG，Label 子目录存放 *_label.PNG
                label_dir = os.path.join(train_dir, 'Label')
                for fn in sorted(os.listdir(train_dir)):
                    fp = os.path.join(train_dir, fn)
                    if not fn.lower().endswith('.png') or not os.path.isfile(fp):
                        continue
                    stem = os.path.splitext(fn)[0]
                    # 查找对应掩码
                    mask_fp = os.path.join(label_dir, f'{stem}_label.PNG')
                    if os.path.exists(mask_fp):
                        self.image_paths.append(fp)
                        self.mask_paths.append(mask_fp)
                        self.labels.append(1)
                    else:
                        self.image_paths.append(fp)
                        self.mask_paths.append(None)
                        self.labels.append(0)
            else:
                # Test 同理
                label_dir = os.path.join(test_dir, 'Label')
                for fn in sorted(os.listdir(test_dir)):
                    fp = os.path.join(test_dir, fn)
                    if not fn.lower().endswith('.png') or not os.path.isfile(fp):
                        continue
                    stem = os.path.splitext(fn)[0]
                    mask_fp = os.path.join(label_dir, f'{stem}_label.PNG')
                    if os.path.exists(mask_fp):
                        self.image_paths.append(fp)
                        self.mask_paths.append(mask_fp)
                        self.labels.append(1)
                    else:
                        self.image_paths.append(fp)
                        self.mask_paths.append(None)
                        self.labels.append(0)
    
    def _init_mpdd(self):
        """初始化MPDD数据集"""
        # 如果提供了category，则路径进入对应子目录
        self.root = os.path.join(self.root_dir, self.category) if self.category else self.root_dir
        
        if self.mode == 'train':
            # 训练集：仅使用正常图片
            img_dir = os.path.join(self.root, 'train', 'good')
            if os.path.isdir(img_dir):
                for f in os.listdir(img_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.image_paths.append(os.path.join(img_dir, f))
            self.image_paths.sort()
            
            # 准备真实缺陷形状掩码
            self.mask_shapes = []
            gt_dir = os.path.join(self.root, 'ground_truth')
            if os.path.isdir(gt_dir):
                for defect_type in os.listdir(gt_dir):
                    defect_dir = os.path.join(gt_dir, defect_type)
                    if os.path.isdir(defect_dir):
                        for f in os.listdir(defect_dir):
                            if f.endswith('.png'):
                                self.mask_shapes.append(os.path.join(defect_dir, f))
                    elif defect_type.endswith('.png'):
                        self.mask_shapes.append(os.path.join(gt_dir, defect_type))
            
            # 准备外部纹理图像路径列表
            self.texture_paths = []
            if self.anomaly_source_dir and os.path.isdir(self.anomaly_source_dir):
                for root, _, files in os.walk(self.anomaly_source_dir):
                    for fname in files:
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            self.texture_paths.append(os.path.join(root, fname))
        else:
            # 测试集：加载正常和异常样本
            test_dir = os.path.join(self.root, 'test')
            if not os.path.isdir(test_dir):
                raise RuntimeError(f"测试目录未找到: {test_dir}")
            
            for defect_type in os.listdir(test_dir):
                ddir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(ddir):
                    continue
                for fname in os.listdir(ddir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and '_mask' not in fname:
                        img_path = os.path.join(ddir, fname)
                        self.image_paths.append(img_path)
                        
                        # 对应的GT掩码路径
                        mask_path = os.path.join(self.root, 'ground_truth', defect_type, 
                                                fname.split('.')[0] + '_mask.png')
                        if not os.path.exists(mask_path):
                            mask_path = None
                        self.mask_paths.append(mask_path)
            
            # 按名称排序
            self.image_paths, self.mask_paths = zip(*sorted(zip(self.image_paths, self.mask_paths)))
    
    def _init_mvtec(self):
        """初始化MVTec数据集"""
        self.cls_dir = os.path.join(self.root_dir, self.category)
        if not os.path.isdir(self.cls_dir):
            raise RuntimeError(f"{self.cls_dir} 不存在，请检查 category")
        
        if self.mode == 'train':
            # 训练模式：只读正常样本路径
            train_good_dir = os.path.join(self.cls_dir, 'train', 'good')
            self.image_paths = sorted([
                os.path.join(train_good_dir, fn) for fn in os.listdir(train_good_dir)
                if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
            
            # 外部纹理（可选）
            self.texture_paths = []
            if self.anomaly_source_dir and os.path.isdir(self.anomaly_source_dir):
                for r, _, fs in os.walk(self.anomaly_source_dir):
                    for fn in fs:
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            self.texture_paths.append(os.path.join(r, fn))
            
            # 预先生成的前景掩码路径列表
            mask_dir = os.path.join(self.cls_dir, 'train_mask', 'good')
            self.mask_paths = []
            for img_path in self.image_paths:
                base, ext = os.path.splitext(os.path.basename(img_path))
                mask_name = f"{base}_mask{ext}"
                mask_path = os.path.join(mask_dir, mask_name)
                if not os.path.isfile(mask_path):
                    # 如果没有预生成掩码，创建全前景掩码
                    self.mask_paths.append(None)
                else:
                    self.mask_paths.append(mask_path)
        else:
            # 测试模式：加载test集下所有图像路径和标签、掩码路径
            test_root = os.path.join(self.cls_dir, 'test')
            gt_root = os.path.join(self.cls_dir, 'ground_truth')
            
            for label_dir in sorted(os.listdir(test_root)):
                dpath = os.path.join(test_root, label_dir)
                if not os.path.isdir(dpath):
                    continue
                for fn in sorted(os.listdir(dpath)):
                    if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    imgp = os.path.join(dpath, fn)
                    self.image_paths.append(imgp)
                    if label_dir == 'good':
                        self.labels.append(0)
                        self.mask_paths.append(None)
                    else:
                        self.labels.append(1)
                        # 测试集mask文件名格式: <原名>_mask<ext>
                        base, ext = os.path.splitext(fn)
                        mp = os.path.join(gt_root, label_dir, f"{base}_mask{ext}")
                        if not os.path.isfile(mp):
                            mp = None
                        self.mask_paths.append(mp)
    
    def _init_visa(self):
        """初始化VisA数据集"""
        cat_dir = os.path.join(self.root_dir, self.category)
        
        if self.mode == 'train':
            # 训练模式：只使用 train/good 中的正常样本
            train_good_dir = os.path.join(cat_dir, 'train', 'good')
            if os.path.exists(train_good_dir):
                for img_name in os.listdir(train_good_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(train_good_dir, img_name)
                        self.image_paths.append(img_path)
            
            # 使用ground_truth/bad中的掩码作为形状模板
            mask_dir = os.path.join(cat_dir, 'ground_truth', 'bad')
            self.mask_shapes = []
            if os.path.exists(mask_dir):
                self.mask_shapes = [
                    os.path.join(mask_dir, fn)
                    for fn in os.listdir(mask_dir)
                    if fn.lower().endswith('.png')
                ]
            
            # 加载前景掩码
            fg_mask_dir = os.path.join(cat_dir, 'train_mask', 'good')
            self.fg_masks = []
            if os.path.exists(fg_mask_dir):
                self.fg_masks = [
                    os.path.join(fg_mask_dir, fn)
                    for fn in os.listdir(fg_mask_dir)
                    if fn.lower().endswith('.png')
                ]
            
            self.texture_paths = []
            if self.anomaly_source_dir and os.path.isdir(self.anomaly_source_dir):
                for r, _, files in os.walk(self.anomaly_source_dir):
                    for fn in files:
                        if fn.lower().endswith(('.png','jpg','jpeg')):
                            self.texture_paths.append(os.path.join(r, fn))
        else:
            # 测试模式：使用 test/good 和 test/bad
            test_good_dir = os.path.join(cat_dir, 'test', 'good')
            test_bad_dir = os.path.join(cat_dir, 'test', 'bad')
            gt_bad_dir = os.path.join(cat_dir, 'ground_truth', 'bad')
            
            # 正常样本
            if os.path.exists(test_good_dir):
                for img_name in os.listdir(test_good_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(test_good_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(0)
                        self.mask_paths.append(None)
            
            # 异常样本
            if os.path.exists(test_bad_dir):
                for img_name in os.listdir(test_bad_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(test_bad_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(1)
                        
                        # 查找对应的掩码文件
                        mask_name = img_name.replace('.JPG', '.png').replace('.jpg', '.png').replace('.jpeg', '.png')
                        mask_path = os.path.join(gt_bad_dir, mask_name)
                        if not os.path.exists(mask_path):
                            mask_path = None
                        self.mask_paths.append(mask_path)
    
    def __len__(self):
        if self.mode == 'train' and self.dataset_type == 'mvtec':
            # MVTec训练模式返回3倍长度（正常、背景异常、结构异常各一次）
            return len(self.image_paths) * 3
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self._get_train_item(idx)
        else:
            return self._get_test_item(idx)
    
    def _get_train_item(self, idx):
        """获取训练样本"""
        if self.dataset_type == 'btad':
            return self._get_btad_train_item(idx)
        elif self.dataset_type == 'dagm':
            return self._get_dagm_train_item(idx)
        elif self.dataset_type == 'mpdd':
            return self._get_mpdd_train_item(idx)
        elif self.dataset_type == 'mvtec':
            return self._get_mvtec_train_item(idx)
        elif self.dataset_type == 'visa':
            return self._get_visa_train_item(idx)
    
    def _get_test_item(self, idx):
        """获取测试样本"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        label = self.labels[idx] if hasattr(self, 'labels') and self.labels else 0
        mask_path = self.mask_paths[idx] if hasattr(self, 'mask_paths') and self.mask_paths else None
        
        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L').resize((self.resize, self.resize))
            mask_tensor = transforms.ToTensor()(mask_img)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, self.resize, self.resize), dtype=torch.float32)
        
        return img_tensor, label, mask_tensor
    
    def _get_btad_train_item(self, idx):
        """BTAD训练样本获取"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.resize, self.resize))
        img_np = np.array(img)
        orig_np = img_np.copy()
        
        h, w, _ = img_np.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if hasattr(self, 'mask_shapes') and self.mask_shapes and hasattr(self, 'texture_paths') and self.texture_paths and random.random() < 0.5:
            # 真实形状 + 纹理
            m = random.choice(self.mask_shapes)
            m_img = Image.open(m).convert('L').resize((w, h))
            m_arr = (np.array(m_img) > 128).astype(np.uint8)
            tex = random.choice(self.texture_paths)
            t_img = Image.open(tex).convert('RGB').resize((w, h))
            t_arr = np.array(t_img)
            img_np[m_arr==1] = t_arr[m_arr==1]
            mask[m_arr==1] = 1
        else:
            # CutPaste
            crop_h = random.randint(h//8, h//3)
            crop_w = random.randint(w//8, w//3)
            x0 = random.randint(0, w-crop_w)
            y0 = random.randint(0, h-crop_h)
            patch = img_np[y0:y0+crop_h, x0:x0+crop_w].copy()
            if random.random() < 0.5: patch = np.flip(patch, axis=1)
            if random.random() < 0.5: patch = np.flip(patch, axis=0)
            factor = random.uniform(0.5,1.5)
            patch = np.clip(patch.astype(np.float32)*factor,0,255).astype(np.uint8)
            x1 = random.randint(0, w-crop_w)
            y1 = random.randint(0, h-crop_h)
            img_np[y1:y1+crop_h, x1:x1+crop_w] = patch
            mask[y1:y1+crop_h, x1:x1+crop_w] = 1
        
        aug_img = Image.fromarray(img_np)
        img_tensor = self.transform(aug_img)
        orig_img = Image.fromarray(orig_np)
        orig_tensor = self.transform(orig_img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img_tensor, orig_tensor, mask_tensor
    
    def _get_dagm_train_item(self, idx):
        """DAGM训练样本获取"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        
        # DAGM训练模式：重构目标即原图
        return img_tensor, img_tensor.clone(), torch.zeros((1, self.resize, self.resize), dtype=torch.float32)
    
    def _get_mpdd_train_item(self, idx):
        """MPDD训练样本获取"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.resize, self.resize))
        img_np = np.array(img)
        orig_np = img_np.copy()
        
        mask = np.zeros((self.resize, self.resize), dtype=np.uint8)
        
        # 随机选择伪缺陷注入策略
        use_external = (random.random() < 0.5 and hasattr(self, 'mask_shapes') and self.mask_shapes 
                       and hasattr(self, 'texture_paths') and self.texture_paths)
        
        if use_external:
            # 使用真实缺陷形状 + 外部纹理
            mask_path = random.choice(self.mask_shapes)
            mask_img = Image.open(mask_path).convert('L').resize((self.resize, self.resize))
            mask_arr = np.array(mask_img)
            mask_bin = (mask_arr > 128).astype(np.uint8)
            
            tex_path = random.choice(self.texture_paths)
            tex_img = Image.open(tex_path).convert('RGB').resize((self.resize, self.resize))
            tex_arr = np.array(tex_img)
            
            img_np[mask_bin == 1] = tex_arr[mask_bin == 1]
            mask[mask_bin == 1] = 1
        else:
            # CutPaste
            h, w, _ = img_np.shape
            crop_h = random.randint(h//8, h//3)
            crop_w = random.randint(w//8, w//3)
            x0 = random.randint(0, w - crop_w)
            y0 = random.randint(0, h - crop_h)
            patch = img_np[y0:y0+crop_h, x0:x0+crop_w].copy()
            
            if random.random() < 0.5:
                patch = np.flip(patch, axis=1)
            if random.random() < 0.5:
                patch = np.flip(patch, axis=0)
            
            factor = random.uniform(0.5, 1.5)
            patch = np.clip(patch.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            
            new_x0, new_y0 = x0, y0
            attempt = 0
            while attempt < 5 and (abs(new_x0 - x0) < crop_w and abs(new_y0 - y0) < crop_h):
                new_x0 = random.randint(0, w - crop_w)
                new_y0 = random.randint(0, h - crop_h)
                attempt += 1
            
            img_np[new_y0:new_y0+crop_h, new_x0:new_x0+crop_w] = patch
            mask[new_y0:new_y0+crop_h, new_x0:new_x0+crop_w] = 1
        
        aug_img = Image.fromarray(img_np)
        img_tensor = self.transform(aug_img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        orig_img = Image.fromarray(orig_np)
        orig_tensor = self.transform(orig_img)
        
        return img_tensor, mask_tensor, orig_tensor
    
    def _get_mvtec_train_item(self, idx):
        """MVTec训练样本获取"""
        N = len(self.image_paths)
        base_idx = idx % N
        type_idx = idx // N
        
        img_path = self.image_paths[base_idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.resize, self.resize))
        img_np = np.array(img)
        orig_np = img_np.copy()
        
        # 读取前景掩码
        obj_mask = np.ones((self.resize, self.resize), dtype=np.uint8)
        if hasattr(self, 'mask_paths') and self.mask_paths and self.mask_paths[base_idx]:
            obj_mask_img = Image.open(self.mask_paths[base_idx]).convert('L').resize((self.resize, self.resize))
            obj_mask = (np.array(obj_mask_img) > 0).astype(np.uint8)
        
        h, w, _ = img_np.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if type_idx == 0:
            # 正常样本
            pass
        elif type_idx == 1:
            # 背景异常样本
            self._apply_background_anomaly(img_np, mask, obj_mask, h, w)
        else:
            # 结构缺陷样本
            self._apply_structural_anomaly(img_np, mask, obj_mask, h, w)
        
        img_tensor = self.transform(Image.fromarray(img_np))
        orig_tensor = self.transform(Image.fromarray(orig_np))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img_tensor, orig_tensor, mask_tensor
    
    def _apply_background_anomaly(self, img_np, mask, obj_mask, h, w):
        """应用背景异常"""
        if hasattr(self, 'texture_paths') and self.texture_paths:
            # 使用外部纹理
            tex_img = Image.open(random.choice(self.texture_paths)).convert('RGB').resize((w, h))
            tnp = np.array(tex_img)
            
            if random.random() < 0.5:
                # 椭圆形区域
                ell_w = random.randint(w // 3, w)
                ell_h = random.randint(h // 3, h)
                x0 = random.randint(0, w - ell_w)
                y0 = random.randint(0, h - ell_h)
                
                mask_img = Image.new('L', (w, h), 0)
                draw = ImageDraw.Draw(mask_img)
                draw.ellipse((x0, y0, x0 + ell_w, y0 + ell_h), fill=255)
                blur_radius = random.randint(10, 30)
                mask_img = mask_img.filter(ImageFilter.GaussianBlur(blur_radius))
                mask_array = np.array(mask_img).astype(np.float32) / 255.0
                
                mask_array[obj_mask == 1] = 0
                mask_blend = mask_array[:, :, None]
                img_np[:] = (img_np * (1 - mask_blend) + tnp * mask_blend).astype(np.uint8)
                defect_region = ((mask_array > 0.5) & (obj_mask == 0))
                mask[defect_region] = 1
            else:
                # 噪声形状区域
                small = random.randint(32, 64)
                noise = np.random.rand(small, small) * 255
                noise_img = Image.fromarray(noise.astype(np.uint8)).resize((w, h), resample=Image.BICUBIC)
                noise_array = np.array(noise_img).astype(np.float32)
                thr = random.uniform(0.4, 0.6) * 255.0
                mask_bin = (noise_array > thr).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_bin, mode='L').filter(ImageFilter.GaussianBlur(5))
                mask_array = np.array(mask_img).astype(np.float32) / 255.0
                
                mask_array[obj_mask == 1] = 0
                mask_blend = mask_array[:, :, None]
                img_np[:] = (img_np * (1 - mask_blend) + tnp * mask_blend).astype(np.uint8)
                defect_region = ((mask_array > 0.5) & (obj_mask == 0))
                mask[defect_region] = 1
    
    def _apply_structural_anomaly(self, img_np, mask, obj_mask, h, w):
        """应用结构异常"""
        if hasattr(self, 'texture_paths') and self.texture_paths and random.random() < 0.5:
            # 外部纹理注入结构缺陷
            tex_img = Image.open(random.choice(self.texture_paths)).convert('RGB').resize((w, h))
            tnp = np.array(tex_img)
            
            ch = random.randint(h // 8, h // 3)
            cw = random.randint(w // 8, w // 3)
            x0 = random.randint(0, w - cw)
            y0 = random.randint(0, h - ch)
            
            for yy in range(ch):
                for xx in range(cw):
                    if obj_mask[y0+yy, x0+xx] == 1:
                        img_np[y0+yy, x0+xx] = tnp[y0+yy, x0+xx]
                        mask[y0+yy, x0+xx] = 1
        else:
            # 使用原图块制造结构缺陷
            ch = random.randint(h // 8, h // 3)
            cw = random.randint(w // 8, w // 3)
            
            max_tries = 5
            for _ in range(max_tries):
                x0 = random.randint(0, w - cw)
                y0 = random.randint(0, h - ch)
                sub_mask = obj_mask[y0:y0+ch, x0:x0+cw]
                if sub_mask.sum() > 0.5 * ch * cw:
                    break
            
            patch = img_np[y0:y0 + ch, x0:x0 + cw].copy()
            
            if random.random() < 0.5:
                patch = np.flip(patch, axis=1)
            if random.random() < 0.5:
                patch = np.flip(patch, axis=0)
            if random.random() < 0.5:
                factor = random.uniform(0.5, 1.5)
                patch = np.clip(patch.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            
            x1 = random.randint(0, w - cw)
            y1 = random.randint(0, h - ch)
            
            for yy in range(ch):
                for xx in range(cw):
                    if obj_mask[y1+yy, x1+xx] == 1:
                        img_np[y1+yy, x1+xx] = patch[yy, xx]
                        mask[y1+yy, x1+xx] = 1
    
    def _get_visa_train_item(self, idx):
        """VisA训练样本获取"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB').resize((self.resize, self.resize))
        img_np = np.array(img)
        orig_np = img_np.copy()
        
        mask = np.zeros((self.resize, self.resize), dtype=np.uint8)
        
        # 加载对应的前景掩码
        fg_mask = np.ones((self.resize, self.resize), dtype=np.uint8)
        if hasattr(self, 'fg_masks') and self.fg_masks:
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace('.JPG', '_mask.png').replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
            for fg_mask_path in self.fg_masks:
                if os.path.basename(fg_mask_path) == mask_name:
                    fg_img = Image.open(fg_mask_path).convert('L').resize((self.resize, self.resize))
                    fg_mask = (np.array(fg_img) > 128).astype(np.uint8)
                    break
        
        if (random.random() < 0.5 and hasattr(self, 'mask_shapes') and self.mask_shapes
            and hasattr(self, 'texture_paths') and self.texture_paths):
            # 真掩码形状 + 外部纹理
            mp = random.choice(self.mask_shapes)
            m_img = Image.open(mp).convert('L').resize((self.resize, self.resize))
            m_bin = (np.array(m_img) > 128).astype(np.uint8)
            m_bin = m_bin * fg_mask
            
            if np.sum(m_bin) > 0:
                tex = Image.open(random.choice(self.texture_paths)).convert('RGB').resize((self.resize, self.resize))
                tex_np = np.array(tex)
                img_np[m_bin==1] = tex_np[m_bin==1]
                mask[m_bin==1] = 1
        else:
            # 自身区域CutPaste
            h, w, _ = img_np.shape
            fg_coords = np.where(fg_mask == 1)
            
            if len(fg_coords[0]) > 0:
                min_y, max_y = fg_coords[0].min(), fg_coords[0].max()
                min_x, max_x = fg_coords[1].min(), fg_coords[1].max()
                
                ch = random.randint(min(h//8, (max_y-min_y)//4), min(h//3, (max_y-min_y)//2))
                cw = random.randint(min(w//8, (max_x-min_x)//4), min(w//3, (max_x-min_x)//2))
                
                if ch > 0 and cw > 0:
                    x0 = random.randint(max(0, min_x), min(w-cw, max_x-cw))
                    y0 = random.randint(max(0, min_y), min(h-ch, max_y-ch))
                    patch = img_np[y0:y0+ch, x0:x0+cw].copy()
                    
                    if random.random() < 0.5: patch = np.flip(patch, axis=1)
                    if random.random() < 0.5: patch = np.flip(patch, axis=0)
                    factor = random.uniform(0.5, 1.5)
                    patch = np.clip(patch.astype(np.float32)*factor, 0,255).astype(np.uint8)
                    
                    nx, ny = x0, y0
                    for _ in range(10):
                        tx = random.randint(max(0, min_x), min(w-cw, max_x-cw))
                        ty = random.randint(max(0, min_y), min(h-ch, max_y-ch))
                        if abs(tx-x0)>cw//2 or abs(ty-y0)>ch//2:
                            nx, ny = tx, ty
                            break
                    
                    patch_mask = fg_mask[ny:ny+ch, nx:nx+cw]
                    img_np[ny:ny+ch, nx:nx+cw][patch_mask==1] = patch[patch_mask==1]
                    mask[ny:ny+ch, nx:nx+cw] = patch_mask
        
        aug_tensor = self.transform(Image.fromarray(img_np))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        orig_tensor = self.transform(Image.fromarray(orig_np))
        
        return aug_tensor, mask_tensor, orig_tensor


# 为了向后兼容，保留原始类名的别名
BTADDataset = lambda *args, **kwargs: UnifiedAnomalyDataset(*args, dataset_type='btad', **kwargs)
DAGMDataset = lambda *args, **kwargs: UnifiedAnomalyDataset(*args, dataset_type='dagm', **kwargs)
MPDDDataset = lambda *args, **kwargs: UnifiedAnomalyDataset(*args, dataset_type='mpdd', **kwargs)
PCBDataset = lambda *args, **kwargs: UnifiedAnomalyDataset(*args, dataset_type='mpdd', **kwargs)  # PCB使用MPDD逻辑
MVTecADDataset = lambda *args, **kwargs: UnifiedAnomalyDataset(*args, dataset_type='mvtec', **kwargs)
VisADataset = lambda *args, **kwargs: UnifiedAnomalyDataset(*args, dataset_type='visa', **kwargs)