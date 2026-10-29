#!/usr/bin/env python3
# Contrastive pretrain + binary segmentation (ResNet50-FPN) with optional COCO->masks build.

import argparse, random, json, shutil
from pathlib import Path
from typing import Optional
import numpy as np, cv2, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm 

# -------- args --------
def get_args():
    p = argparse.ArgumentParser("Strawberry health segmentation")
    p.add_argument("--root", type=Path, default=Path("/home/ehr/Desktop/anupama/plant/data"))
    p.add_argument("--coco", type=Path, default=Path("/home/ehr/Desktop/anupama/CPP_Strawberry_RGB_MS_2025-26/Unprocessed_data/all.v1i.coco"))
    p.add_argument("--img", type=int, default=512) #IMAgE SIZE
    p.add_argument("--bs", type=int, default=4) #BATCH SIZE BY DEFAULT 4
    p.add_argument("--ep_pre", type=int, default=10) #PRETRAINING EPOCH 10
    p.add_argument("--ep_ft", type=int, default=15) # FINETUNING EPOCH 15
    p.add_argument("--lr_pre", type=float, default=1e-3)  # 1 RAISE TO 3 LEARNING RATE
    p.add_argument("--lr_ft", type=float, default=7e-4) #7 RAISE TO 4 LEARNING RATE
    p.add_argument("--temp", type=float, default=0.07) #TEMPERATURE IS HOW SHARP SIMILARITY DISTRIBUTION
    p.add_argument("--down", type=int, default=64) #DOWNSAMPLE CONTRASTIVE EMBEDDING MAP
    p.add_argument("--posw", type=float, default=3.0) #POSITIVE CLAS WEIGHT FOR BINARY CROSS ENTROPY
    p.add_argument("--seed", type=int, default=42) #RANDOM SEED ENSURES SHUFFLING, AUGMENTATION
    return p.parse_args() #Reads command and fills them into object

A = get_args()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(A.seed); np.random.seed(A.seed); torch.manual_seed(A.seed) #sets the random seed for random, seed and torch liberary

# -------- COCO -> binary PNG masks (optional) --------
def build_from_coco(src: Optional[Path], out: Path): #a function named build
    if src is None: return # src is  the source dataset directory
    print(f"→ Building processed dataset from COCO: {src} → {out}") #out is the destination directory where masked images are saved
    try:
        from pycocotools import mask as R #tries to mask handling utilities
    except Exception:
        R = None; print("⚠ pycocotools not found (RLE masks skipped). pip install pycocotools")  #if source is none, the function would do nothing.

    def poly(seg,h,w): #creates an empty black mask (all zeros)
        m=np.zeros((h,w),np.uint8) #suitable for saving as image later
        for s in seg: #loops over each polygon in segmentation list
            pts=np.asarray(s,np.float32).reshape(-1,2).round().astype(np.int32) #converts the list into numpy array
            if len(pts)>=3: cv2.fillPoly(m,[pts],1) #polygon must have at least 3 points
        return m  #returns the final binary mask array (0 for background, 1 for foreground)

    def rle(seg,h,w): #turning RLE COCO into binary
        if R is None: return np.zeros((h,w),np.uint8) #if protocol was not imported earlier , can not decode RLE. Return all zero mask instead of failing.
        r = R.frPyObjects(seg,h,w) if isinstance(seg.get("counts",None),list) else seg  # dictionary would be used as it is, list needs conversion
        m = R.decode(r); return (m if m.ndim==2 else m.any(2)).astype(np.uint8) # turns RLE into mask arrays.

    def bbox(b,h,w): #defines a helper to convert a bounding box to binary mask
        x,y,bw,bh=b; x1,y1=max(0,int(x)),max(0,int(y)); x2,y2=min(w,int(x+bw)),min(h,int(y+bh)) #coco bbox format is x,y, width, height
        m=np.zeros((h,w),np.uint8); #create empty mask
        if x2>x1 and y2>y1: m[y1:y2,x1:x2]=1 #if bobox is valid fill that rectangle with ones
        return m #return the bbox mask

    for split in ["train","valid","test"]: #Process three dataset splits if they exist in your COCO directory.
        ann=src/split/"_annotations.coco.json" # coco path
        if not ann.exists(): print(f"skip {split}: no {ann}"); continue #if json file not there skip the split
        coco=json.loads(ann.read_text()) #load json into a python dictionary
        imgs={im["id"]:im["file_name"] for im in coco.get("images",[])} # build a dict mapping for quick lookup
        by={}
        for a in coco.get("annotations",[]): by.setdefault(a["image_id"],[]).append(a) #group annotations by image id so we can process all instances for each image

        out_split = "val" if split=="valid" else split #normalise valid as val
        (out/out_split/"images").mkdir(parents=True, exist_ok=True) #ensure output directories exists
        (out/out_split/"masks").mkdir(parents=True,  exist_ok=True)

        for iid,f in tqdm(imgs.items(),desc=f"build:{split}"): #loop through each imageid , filename
            src_img=next((p for p in [src/split/f, src/split/"images"/f] if p.exists()), None)#find actual image under disk
            if src_img is None: continue #if the image file missing, skip this entry
            im=cv2.imread(str(src_img)) #read the image to get the shape
            if im is None: continue #if open cv fails to read , skip
            h,w=im.shape[:2]; M=np.zeros((h,w),np.uint8) #grab image height, width,initialise a composite maskM with zeros
            for a in by.get(iid,[]): #for each annotation of image , fetch its segmentation
                seg,b=a.get("segmentation"),a.get("bbox")
                if isinstance(seg,list) and seg: inst=poly(seg,h,w) #if seg is list and non empty , build a polygon mask with poly
                elif isinstance(seg,dict) and "counts" in seg: inst=rle(seg,h,w)  #if seg is dic with counts,decode using RLE
                elif b is not None: inst=bbox(b,h,w) #If there is bbox,use bbox fallback
                else: continue #if no usable annotation, skip this instance
                M[inst>0]=1  #union instance mask into composite mask, any positive pixels set m to 1
            Image.fromarray(M*255).save(out/out_split/"masks"/(Path(f).stem+".png")) # convert binary mask greyscale  and save as png with the same stem as the image file.
            shutil.copy2(src_img, out/out_split/"images"/Path(f).name) #copy the original image into processed image folder
    print(f"✓ Processed dataset built at {out}")

# -------- sanity check (ensure pairs; auto-build if needed) --------
def sanity_or_build(root: Path, coco: Optional[Path], split="train"):
    img_dir, msk_dir = root/split/"images", root/split/"masks"
    imgs = sorted([p for p in img_dir.glob("*") if p.is_file()])
    pairs = [(p, list(msk_dir.glob(p.stem + ".*"))) for p in imgs]
    ok = [(p, m[0]) for p, m in pairs if m]
    if ok: return
    if coco is not None:
        print(f"⚠ No pairs in {img_dir} & {msk_dir}. Attempting build from COCO…")
        build_from_coco(coco, root)
        imgs = sorted([p for p in img_dir.glob("*") if p.is_file()])
        pairs = [(p, list(msk_dir.glob(p.stem + ".*"))) for p in imgs]
        if [(p, m[0]) for p, m in pairs if m]: print("✓ Pairs found after build."); return
    nI, nM = len(list(img_dir.glob("*"))), len(list(msk_dir.glob("*")))
    someI = [p.stem for p in imgs[:5]]
    someM = [p.stem for p in sorted(msk_dir.glob("*"))[:5]]
    raise RuntimeError(f"No image/mask pairs in {img_dir} & {msk_dir}\nimages: {nI}, masks: {nM}\nsample image stems: {someI}\nsample mask stems : {someM}\nStems must match (IMG_001.jpg ↔ IMG_001.png).")

# -------- data --------
class SegBin(Dataset): # define a custom pytorch dataset segbin short binary segmentation
    def __init__(self, root, split, size=512, aug=False): #base dataset directory, which subset to load
        r=Path(root); I=r/split/"images"; M=r/split/"masks" #convert root to a path object for easy path joining.
        self.size, self.aug = size, aug # store size and aug as instance variables for later use
        self.tt=transforms.ToTensor() #define a transform that convert PIL image into pytorch tensor
        self.cj=transforms.ColorJitter(0.3,0.3,0.2,0.1) if aug else None #If aug=True, create a color jitter transformation (random changes to brightness, contrast, saturation, and hue) to improve model generalization.
        imgs=sorted([p for p in I.glob("*") if p.is_file()]) # list all image file into an image directory, scan and filterout directory
        self.pairs=[(p, list(M.glob(p.stem+".*"))[0]) for p in imgs if list(M.glob(p.stem+".*"))] #build a list of image path, mask path pairs
        if not self.pairs: raise RuntimeError(f"No image/mask pairs in {I} & {M}") # build a list of image pair
    def __len__(self): return len(self.pairs)  #number of sample in the dataset
    def __getitem__(self,i): # define how to load and return one sample
        ip,mp=self.pairs[i] # reterieves ith image and mask path
        img=np.array(Image.open(ip).convert("RGB")) # open the image file using PIL and ensure its RGB
        msk=np.array(Image.open(mp).convert("L")) # Open the mask file and convert it to greyscale
        if self.aug: #if augmentation are enabled
            if random.random()<0.5: img,msk=img[:,::-1].copy(),msk[:,::-1].copy() # with 50% probability ,perform a horizontal flip on both image and mask.
            img=np.array(self.cj(Image.fromarray(img))) # applies the color jitter transformation on the image only
        img=cv2.resize(img,(A.img,A.img)) #Resizes the image to (A.img, A.img) pixels (e.g., 512×512) using OpenCV’s bilinear interpolation (default).
        msk=cv2.resize(msk,(A.img,A.img),interpolation=cv2.INTER_NEAREST) # Resizes the mask to the same size but uses nearest-neighbor interpolation (important!).
        return self.tt(Image.fromarray(img)), torch.from_numpy((msk>0).astype(np.float32))[None] #tensor for image input/output

def loaders(aug=False): # define a helper function called loader
    tr=SegBin(A.root,"train",A.img,aug); va=SegBin(A.root,"val",A.img,False)  #create a training dataset using your custom segbin class
    return (DataLoader(tr,A.bs,True,num_workers=4,pin_memory=True), # create the validation dataset
            DataLoader(va,A.bs,False,num_workers=4,pin_memory=True)) # wrap both dataset in pytorch dataloader for batching and fast loading

# -------- model --------
class Encoder(nn.Module): #define an encoder class for extracting visual  features
    def __init__(self): #initialise the class
        super().__init__()
        m=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) #loads a resnet 50 model
        self.stem=nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool) #define the stem (first few layers)
        self.l1,self.l2,self.l3,self.l4=m.layer1,m.layer2,m.layer3,m.layer4 #grabs the 4 main residual blocks from resnet 50
    def forward(self,x): # input image goes through each layer
        x=self.stem(x); c1=self.l1(x); c2=self.l2(c1); c3=self.l3(c2); c4=self.l4(c3) #extract features from 4 stages:c1,c2,c3,c4
        return (c1,c2,c3,c4)

class SegHead(nn.Module):  # define segmentation feature that converts resnet features into masks
    def __init__(self): #initialise the segmentation head
        super().__init__()
        self.l4, self.l3, self.l2, self.l1 = nn.Conv2d(2048,256,1), nn.Conv2d(1024,256,1), nn.Conv2d(512,256,1), nn.Conv2d(256,256,1)
        #Creates four 1×1 convolution layers (reduce channels to 256 each).
        self.out=nn.Conv2d(256,1,1) #Final output layer: a 1×1 convolution that reduces the 256 channels to 1 output channel (binary mask).
    def forward(self,fs):
        c1,c2,c3,c4=fs #forward fuction takes tuple produced by encoder
        p4=self.l4(c4) #peocessed the deepest feature c4 through 1×1 conv
        p3=self.l3(c3)+F.interpolate(p4,size=c3.shape[-2:],mode="bilinear",align_corners=False) #Upsample p4 to match the size of c3 and add it to the processed c3.
        p2=self.l2(c2)+F.interpolate(p3,size=c2.shape[-2:],mode="bilinear",align_corners=False)  #upsample and add to c2 — going from coarse to finer detail.
        p1=self.l1(c1)+F.interpolate(p2,size=c1.shape[-2:],mode="bilinear",align_corners=False) #Continue the top-down process — merge with the lowest-level features (c1).
        return self.out(F.interpolate(p1,scale_factor=4,mode="bilinear",align_corners=False)) #Upsample p1 by ×4 to reach the original image resolution.

class Proj(nn.Module):  # contrastive projector (defines projection head for contrastive pretraining )
    def __init__(self,c=2048,h=256,o=128): # constructor initialises layer sizes
        super().__init__()  #calls the parent class initializer
        self.net=nn.Sequential(nn.Conv2d(c,h,1,bias=False),nn.BatchNorm2d(h),nn.ReLU(True),nn.Conv2d(h,o,1,bias=False))
    def forward(self,x): return F.normalize(self.net(x),1) #run input x through the network

# -------- losses/metrics -------- #
def dice_loss(log,y,eps=1e-6): #define the dice loss commonly used for segmentation
    p=torch.sigmoid(log); num=2*(p*y).sum((2,3))+eps; den=p.sum((2,3))+y.sum((2,3))+eps #Converts logits to probabilities between 0–1 using the sigmoid function.
    return 1-(num/den).mean() #Computes Dice loss = 1 - mean(Dice coefficient).

def supcon_within(z,y,T): # Computes Supervised Contrastive Loss within each image.
    B,C,h,w=z.shape; z=z.permute(0,2,3,1).reshape(B,h*w,C); y=y.reshape(B,h*w)
    loss=0.0;n=0; eye=torch.eye(h*w,device=z.device,dtype=torch.bool) #initialize total loss and counter for valid batches
    for b in range(B): #loop through each image in batch
        zb,yb=z[b],y[b]; sim=(zb@zb.T)/T; sim-=sim.max(1,keepdim=True).values.detach() # get embedding zb and labels yb for one image.
        exp=torch.exp(sim); same=(yb[:,None]==yb[None,:])&(~eye) #xponentiate similarities for softmax computation., Create a boolean matrix where entries are True if two pixels belong to the same class but not the same pixel.
        denom=(exp*(~eye)).sum(1)+1e-8; pos=(exp*same).sum(1)+1e-8; valid=same.sum(1)>0 #Denominator = sum of exponentials of all other pixels.
        if valid.any(): loss+=(-torch.log(pos/denom))[valid].mean(); n+=1 #Compute -log(pos/denom) for valid pixels (contrastive loss).
    return loss/max(n,1) #return the average loss over valid images.

def metrics(enc,seg,dl,thr=0.5): #computes evaluation matrix over dataloader
    enc.eval(); seg.eval(); P=R=F1=IoU=0; n=0 #set both model to evaluation mode
    with torch.no_grad(): # Disables gradient calculation to save memory during evaluation.
        for x,y in dl: #Loop through batches from the dataloader.
            x,y=x.to(DEVICE),y.to(DEVICE)  # move data to GPU or CPU
            log=F.interpolate(seg(enc(x)),size=y.shape[-2:],mode="bilinear",align_corners=False) #Upsample prediction to match target mask size.
            p=(torch.sigmoid(log)>thr).float() #Apply sigmoid to get probabilities, threshold to get binary mask.
            tp=(y*p).sum(); fp=((1-y)*p).sum(); fn=(y*(1-p)).sum() #Compute true positives, false positives, and false negatives.
            prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8) #Compute precision and recall safely (add epsilon to avoid division by zero).
            f1=2*prec*rec/(prec+rec+1e-8); iou=tp/(tp+fp+fn+1e-8) #Compute F1 score and IoU (Intersection over Union).
            P+=prec.item(); R+=rec.item(); F1+=f1.item(); IoU+=iou.item(); n+=1 #Accumulate metrics for each batch
    return P/n,R/n,F1/n,IoU/n #Return average Precision, Recall, F1, and IoU across all batches.

# -------- training --------
def loaders_train(aug_pre=False): #Creates training and validation dataloaders.
    tr=SegBin(A.root,"train",A.img,aug_pre); va=SegBin(A.root,"val",A.img,False) #Load train and validation datasets.
    return (DataLoader(tr,A.bs,True,num_workers=4,pin_memory=True),
            DataLoader(va,A.bs,False,num_workers=4,pin_memory=True)) #Wraps them in DataLoaders for batching.

def pretrain(): #Defines the pretraining function.
    dl,_=loaders_train(True); enc,prj=Encoder().to(DEVICE),Proj().to(DEVICE) #Get training dataloader with augmentation ,Initialize encoder and projector models on the device.
    opt=torch.optim.AdamW(list(enc.parameters())+list(prj.parameters()),lr=A.lr_pre,weight_decay=1e-4) #Use AdamW optimizer for both models with learning rate A.lr_pre.
    for e in range(1,A.ep_pre+1): #Loop through pretraining epochs.
        enc.train(); prj.train(); pbar=tqdm(dl,desc=f"[Pretrain {e}/{A.ep_pre}]"); meter=n=0 #Set models to training mode and create a progress bar.
        for x,y in pbar:
            x,y=x.to(DEVICE),y.to(DEVICE) #For each batch, move data to GPU/CPU.
            z=prj(F.interpolate(enc(x)[-1],(A.down,A.down),mode="bilinear",align_corners=False)) #Extract the deepest encoder features
            loss=supcon_within(z,F.interpolate(y,(A.down,A.down),mode="nearest")[:,0].long(),A.temp) #Compute supervised contrastive loss with downsampled masks and temperature
            opt.zero_grad(); loss.backward(); opt.step() #Standard training step: zero gradients, backprop, optimizer update.
            meter+=loss.item()*x.size(0); n+=x.size(0); pbar.set_postfix(loss=f"{meter/max(n,1):.4f}") #Track running average loss in progress bar.
    torch.save(enc.state_dict(),"encoder_contrastive.pt"); print("✓ Saved encoder_contrastive.pt") #Save encoder weights after pretraining — used later in fine-tuning.

def finetune(): # Train the full segmentation network with labeled masks.
    tr,va=loaders_train(False); enc,seg=Encoder().to(DEVICE),SegHead().to(DEVICE)
    if Path("encoder_contrastive.pt").exists():
        enc.load_state_dict(torch.load("encoder_contrastive.pt",map_location=DEVICE)); print("Loaded contrastive encoder.")
    opt=torch.optim.AdamW(list(enc.parameters())+list(seg.parameters()),lr=A.lr_ft,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=A.ep_ft)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([A.posw],device=DEVICE))
    best=9e9
    for e in range(1,A.ep_ft+1):
        enc.train(); seg.train(); meter=n=0
        for x,y in tqdm(tr,desc=f"[Finetune {e}/{A.ep_ft}]"):
            x,y=x.to(DEVICE),y.to(DEVICE)
            log=F.interpolate(seg(enc(x)),size=y.shape[-2:],mode="bilinear",align_corners=False)
            loss=0.5*bce(log,y)+0.5*dice_loss(log,y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            meter+=loss.item()*x.size(0); n+=x.size(0)
        # validate
        val_loss=m=0; enc.eval(); seg.eval()
        with torch.no_grad():
            for x,y in va:
                x,y=x.to(DEVICE),y.to(DEVICE)
                log=F.interpolate(seg(enc(x)),size=y.shape[-2:])
                val_loss+=(0.5*bce(log,y)+0.5*dice_loss(log,y)).item()*x.size(0); m+=x.size(0)
        P,R,F1,IoU=metrics(enc,seg,va,thr=0.5)
        print(f"Epoch {e}: train={meter/max(n,1):.4f} val={val_loss/max(m,1):.4f} | P={P:.3f} R={R:.3f} F1={F1:.3f} IoU={IoU:.3f}")
        if val_loss<best:
            best=val_loss; torch.save({"enc":enc.state_dict(),"seg":seg.state_dict()},"plant_unet_contrastive.pt")
            print("✓ Saved plant_unet_contrastive.pt")

# -------- run --------
if __name__=="__main__":
    for sp in ["train","val"]: sanity_or_build(A.root, A.coco, split=sp)
    pretrain(); finetune()
