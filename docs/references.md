# Tài liệu tham khảo đã kiểm chứng (chuẩn IEEE)

Danh sách khởi đầu — chỉ gồm paper đã bình duyệt, sách chuyên khảo và tài liệu chính thức của thư viện. Mỗi mục đều có link gốc để kiểm tra lại.
Thêm tài liệu mới: chỉ lấy từ hội nghị/tạp chí (CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, ACL, IEEE/ACM journals, TMLR...), arXiv của tác giả gốc, hoặc tài liệu chính thức (PyTorch, torchvision). Không dùng blog, bài tổng hợp hay nội dung do AI tạo làm nguồn trích dẫn.

## Paper gốc của đề tài
[1] A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy
[2] A. Vaswani *et al.*, "Attention is all you need," in *Adv. Neural Inf. Process. Syst.*, vol. 30, 2017, pp. 5998–6008. [Online]. Available: https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need
[3] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and dogs," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Providence, RI, USA, 2012, pp. 3498–3505, doi: 10.1109/CVPR.2012.6248092.

## Thành phần của ViT & huấn luyện
[4] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," 2016, *arXiv:1607.06450*.
[5] D. Hendrycks and K. Gimpel, "Gaussian error linear units (GELUs)," 2016, *arXiv:1606.08415*.
[6] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.
[7] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, "Training data-efficient image transformers & distillation through attention," in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, PMLR, vol. 139, 2021, pp. 10347–10357.
[8] A. Steiner *et al.*, "How to train your ViT? Data, augmentation, and regularization in vision transformers," *Trans. Mach. Learn. Res.*, 2022, *arXiv:2106.10270*.

## CNN & transfer learning (nền so sánh, khớp slide 04–07)
[9] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.
[10] S. Kornblith, J. Shlens, and Q. V. Le, "Do better ImageNet models transfer better?," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 2661–2671.
[11] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?," in *Adv. Neural Inf. Process. Syst.*, vol. 27, 2014, pp. 3320–3328.
[12] O. Russakovsky *et al.*, "ImageNet large scale visual recognition challenge," *Int. J. Comput. Vis.*, vol. 115, no. 3, pp. 211–252, 2015.

## Diễn giải mô hình
[13] S. Abnar and W. Zuidema, "Quantifying attention flow in transformers," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2020, pp. 4190–4197, doi: 10.18653/v1/2020.acl-main.385.

## Sách & tài liệu chính thức
[14] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.
[15] A. Zhang, Z. C. Lipton, M. Li, and A. J. Smola, *Dive into Deep Learning*. Cambridge, U.K.: Cambridge Univ. Press, 2023. [Online]. Available: https://d2l.ai
[16] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst.*, vol. 32, 2019, pp. 8024–8035.
[17] PyTorch Contributors. "vit_b_16 — Torchvision documentation." PyTorch. Accessed: Nov. 1, 2026. [Online]. Available: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html
[18] PyTorch Contributors. "OxfordIIITPet — Torchvision documentation." PyTorch. Accessed: Nov. 1, 2026. [Online]. Available: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html

## Chuẩn trình bày dữ liệu / mô hình / repo (dùng cho data card, model card, README)
[19] M. Mitchell *et al.*, "Model cards for model reporting," in *Proc. Conf. Fairness, Accountability, Transparency (FAT\*)*, 2019, pp. 220–229, doi: 10.1145/3287560.3287596.
[20] T. Gebru *et al.*, "Datasheets for datasets," *Commun. ACM*, vol. 64, no. 12, pp. 86–92, Dec. 2021, doi: 10.1145/3458723.
[21] Papers with Code. "Tips for publishing research code (ML Code Completeness Checklist)." GitHub. Accessed: Nov. 1, 2026. [Online]. Available: https://github.com/paperswithcode/releasing-research-code

## Số liệu đối chiếu hay dùng (đã kiểm tra trong nguồn)
- Oxford-IIIT Pet: 37 lớp, `trainval` 3.680 ảnh, `test` 3.669 ảnh [3], [18].
- ViT-B/16: 12 layer, D = 768, MLP 3072, 12 head, ~86M tham số ([1], Table 1); bản torchvision `IMAGENET1K_V1`: 86.567.656 tham số, ImageNet top-1 81,072 %, tiền xử lý resize 256 → crop 224 [17].
- Kết quả paper trên Oxford-IIIT Pets, ViT-B/16 pretrained ImageNet: 93,81 %; pretrained ImageNet-21k: 94,43 % (fine-tune ở 384×384, [1], Table 5).
