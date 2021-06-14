clear;clc;
%psnr=0;ssim=0;
mpsnr=mean(psnr);
mssim=mean(ssim);
%x=3:3:30;
x=0:0.1:0.3;
vp=spcrv([[x(1) x x(end)];[mpsnr(1) mpsnr mpsnr(end)]],3);
vs=spcrv([[x(1) x x(end)];[mssim(1) mssim mssim(end)]],3);
subplot(2,1,1);
plot(vp(1,:),vp(2,:),'k');
title('Average PSNR');
set(gca,'xtick',[0 0.1 0.2 0.3]);
set(gca,'xticklabel',[10^(-3) 10^(-2) 10^(-1) 1]);
xlabel('tau');ylabel('PSNR');
subplot(2,1,2);
plot(vs(1,:),vs(2,:),'k');
title('Average SSIM');
set(gca,'ylim',[0.9663 0.9668]);
set(gca,'xtick',[0 0.1 0.2 0.3]);
set(gca,'xticklabel',[10^(-3) 10^(-2) 10^(-1) 1]);
xlabel('tau');ylabel('SSIM');


save('psnr_ssimt') ;
