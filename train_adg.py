import os
import time

import torch
import torch.distributed as dist
from ceyin.utils.util_list import gen_config_from_parse
from ceyin.utils.visualizer import Visualizer

from ceyin import create_model
from ceyin.controller import Controller
from ceyin.networks import init_net
from ceyin.utils.utility import valid_metrics_cal, init_dataset, random_seed_initial
from options.train_options import TrainOptions

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['NCCL_IB_DISABLE'] = '1'


def main():
    # initial process
    opt = TrainOptions().parse()
    # initialize random seed
    # pygame.init()
    random_seed_initial(opt.seed)

    if opt.model in ['Erase', 'erasenet', 'erase', 'gateconv']:
        opt.data_norm = False

    # device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    device = 'cuda' if opt.gpu_ids else "cpu"
    opt.real_val = False

    # 强化学习训练的gt是reward的,realistic_reward:
    # 对于产生的这一系列组合要给予一个reward
    realistic_reward = False
    realistic_val_reward = False  # 训练集上的合成文本与目标分布收集的验证集上的原始文本之间的差距
    difficult_reward = False

    if "1" in opt.reward_type:
        realistic_reward = True
        opt.netD_M = True
    if "2" in opt.reward_type:
        difficult_reward = True
    if "4" in opt.reward_type:
        realistic_val_reward = True
        opt.netD_M = True  # netD_M 是控制器的判别器
    reward_norm = opt.reward_norm

    # ---------------- 数据加载 ------------------------------------------------------
    dataset, dataset_size = init_dataset(opt, dist)
    dataset.dataset.gen_tr = True  # ？？？？ 不知道是干嘛的，给了True以后在item中的表现就不一样了
    print('#training images = %d' % dataset_size)

    if opt.valid == 1:
        valid_dataset, v_l = init_dataset(opt, dist, batchSize=4, valid=3, serial_batches=True)  # 验证集
        print('#valid images = %d' % v_l)
        test_dataset, t_l = init_dataset(opt, dist, batchSize=4, valid=2, serial_batches=True)  # 测试集
        print('#test images = %d' % t_l)

    # 加载的训练集，给controller用的
    ctl_dataset, ct_1 = init_dataset(opt, dist, batchSize=opt.ctl_batchSize)
    print('#controller train images= %d' % ct_1)

    if realistic_val_reward:  # 加载的验证数据集
        real_dataset, r_l = init_dataset(opt, dist, real_val=True)
        print('#real/val train images = %d' % r_l)

    # -------------------- 模型设计 ------------------------------------------------------------
    # model initial
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    # controller initial
    controller = Controller(gen_space=opt.gen_space, is_cuda=True)
    controller = init_net(controller, opt.init_type, opt.init_gain, opt.gpu_ids, opt.online)
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=opt.clr)

    # ------------------- 记载以训练模型 ---------------------------------
    if opt.continue_train and not opt.adg_start:
        load_filename = '%s_net_Controller.pth' % str(opt.which_epoch)
        load_path = os.path.join(opt.checkpoints_dir, opt.model, opt.name, load_filename)
        state_dict = torch.load(load_path, map_location="cpu")
        controller.load_state_dict(state_dict)

    # begin epoch set
    total_steps = 0
    if opt.continue_train:
        begin_epoch = int(opt.which_epoch) + 1
    else:
        begin_epoch = 1

    if opt.valid != 0 and opt.continue_train:
        _, psnr_v, _, psnr_v_m = valid_metrics_cal(valid_dataset, v_l, model, visualizer, begin_epoch - 1, False, m_type=1)
        _, psnr_t, _, psnr_t_m = valid_metrics_cal(test_dataset, t_l, model, visualizer, begin_epoch - 1, False, m_type=1)
        visualizer.print_valid_metric_list(begin_epoch - 1, [psnr_v, psnr_t, psnr_v_m, psnr_t_m])

    for epoch in range(begin_epoch):
        model.update_learning_rate()

    # -------------------- 控制器参数 ---------------
    best_metrics = 0
    stop_epoch = 0

    M = opt.ctl_M  #
    controller_update = opt.ctl_freq  # 控制器更新的频率

    if difficult_reward:
        # Lm_mean = torch.zeros(1, requires_grad=False).cuda()
        Lm_mean = torch.zeros(1, requires_grad=False).to(device)
        lambda1 = opt.lambda1

    if realistic_reward or realistic_val_reward:
        # Ln_mean = torch.zeros(1, requires_grad=False).cuda() * 0.5
        Ln_mean = torch.zeros(1, requires_grad=False).to(device) * 0.5
        lambda2 = opt.lambda2

    # ------------------------------ 模型训练 --------------------------------------
    for epoch in range(begin_epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()
        epoch_start_time = time.time()
        iter_start_time = time.time()

        # 切成train模式
        model.train()
        controller.train()

        # cost_time = [0 for _ in range(6)]
        p_flag = True

        # ----------------------------------------------------文字擦除模型训练----------------------------------------------------------
        # 一个epoch一个完整的dataset，这里的写法和传统的训练的写法有些差异
        # print(len(dataset))
        for i in range(len(dataset)):  # dataset是数据的个数，这个是根据bs动态调整的
            # cost_time[0] -= time.time()
            # ----------------------------- 每过多少iter产生一次新的policies ------------------------------------
            # 是推理过程
            if i % opt.ctl_train_freq == 0:
                policies, _, _ = controller(batch_size=opt.ctl_policy_num)  # 8,opt.ctl_policy_num:2，
                policies = policies.cpu().detach().numpy()  # policies：2,18
                gen_configs = gen_config_from_parse(policies, opt.gen_space)  # 8
                if p_flag:
                    visualizer.print_policy_log(epoch, gen_configs)
                    p_flag = False
                dataset.dataset.gen_configs = gen_configs  # 在这一步把configs模型的组合传递过去了，伴随着bs产生的一个polices
                dataset_iter = iter(dataset)  # dataset变成一个迭代器了

            # cost_time[0] += time.time()
            # cost_time[1] -= time.time()
            data = next(dataset_iter)  # 每次一个bs的数据
            total_steps += 1  # 和iter不同步的，是整体的step，以iter为单位

            model.set_inputs(data)  # 阿里的代码想喜欢这么写，正常的数据输入流，总是喜欢通过这种方式灌入到model中去
            model.set_specific_image(0)

            if opt.domain_in:
                model.set_basic_input(data)
                model.forward_basic()
                model.optimize_parameters("domain")
            elif opt.baseline == "dann":
                model.optimize_parameters(opt.baseline)
            else:
                model.optimize_parameters()

            # -------------------------------------- controller的前向reward更新 -------------------------------------
            # cost_time[1] += time.time()
            # ------------------------------ controller 的更新是以全局iter做单位的 ----------------------------
            if (total_steps + 1) % controller_update == 0:  # 在total_steps中每隔10轮训练一波
#                 print("enter controller updater")
                # cost_time[2] -= time.time()
                if realistic_val_reward:  # 在内部训练了一个reslistic判别器
                    model.update_val_real_dis(real_dataset, gen_configs)
                # cost_time[2] += time.time()

                # 文字擦除模型 model
                model.eval()
                ctl_batch_num = max(1, int(opt.ctl_update_num / opt.ctl_batchSize))
                # opt.ctl_update_num:6, ctl_bs:2,controller的前向的轮数，注意controller的训练是在model的前向基础上
                for _ in range(ctl_batch_num):
                    # cost_time[3] -= time.time()

                    # 产生M组策略
                    if difficult_reward:  # difficult reward Lmean:历史损失的指数平均数
                        Lm = torch.zeros(M, requires_grad=False).cuda()
#                         Lm = torch.zeros(M*2, requires_grad=False).to(device)
                    if realistic_reward or realistic_val_reward:
                        Ln = torch.zeros(M, requires_grad=False).cuda()
#                         Ln = torch.zeros(M*2, requires_grad=False).to(device)
                    Ls = torch.zeros(M, requires_grad=False).cuda()
#                     Ls = torch.zeros(M*2, requires_grad=False).to(device)

                    policies, log_probs, entropies = controller(batch_size=M)  #
                    policies = policies.cpu().detach().numpy()
                    gen_configs = gen_config_from_parse(policies, opt.gen_space)
                    # print(gen_configs)
                    ctl_dataset.dataset.gen_configs = gen_configs
                    ctl_dataset_iter = iter(ctl_dataset)  # ctl_dataset是train数据集

                    reward_data = next(ctl_dataset_iter)  # 一个bs的拿数据
                    model.set_inputs(reward_data)
                    for k in range(len(gen_configs)):  # 产生的策略数对齐，此处产生了三组策略
                        model.set_specific_image(k)
                        model.forward()
                        if difficult_reward:
                            Lm[k] += model.calcu_loss().sum().detach()
                        if realistic_reward or realistic_val_reward:
                            Ln[k] += model.calcu_real_reward().mean().detach()
                    # cost_time[3] += time.time()

                    # cost_time[4] -= time.time()
                    if difficult_reward:
                        Lm = Lm / opt.ctl_batchSize
                        # print(Lm, lambda1)
                        if Lm_mean == 0:
                            Lm_mean = torch.mean(Lm)
                        Lm_mean = 0.95 * Lm_mean + 0.05 * torch.mean(Lm)
                        if reward_norm == "mean":
                            Lm = (Lm - Lm_mean) * 60
                        elif reward_norm == "exp":
                            # Lm = -torch.abs(1-torch.exp((Lm - 3*Lm_mean)*300))
                            Lm = 5 * (1 - torch.abs(1 - torch.exp((Lm - opt.diff_range * Lm_mean) * 20)))
                        elif reward_norm == "norm":
                            # Lm = 5*(1-torch.abs(1-torch.exp((Lm - opt.diff_range*Lm_mean)*20)))
                            Lm = (Lm - Lm_mean) * 100
                            Lm = (Lm - torch.mean(Lm)) / (torch.std(Lm) + 1e-5)
                        Ls += lambda1 * Lm
                        # print("diffcult reward:", Lm_mean, lambda1 * Lm)

                    if realistic_reward or realistic_val_reward:
                        Ln = Ln / opt.ctl_batchSize
                        Ln_mean = 0.95 * Ln_mean + 0.05 * torch.mean(Ln)
                        if reward_norm == "norm":
                            Ln = -(Ln - torch.mean(Ln)) / (torch.std(Ln) + 1e-5)
                        else:
                            Ln = -10 * (Ln - Ln_mean)
                        Ls += lambda2 * Ln
                        # print("realistic reward:", Ln_mean, lambda2 * Ln)
                    # print("total reward: ", Ls)
                    # cost_time[4] += time.time()

                    # 训练controller，损失函数就是两个reward，一个difficult reward，一个realistic reward的
                    # cost_time[5] -= time.time()
                    controller_optimizer.zero_grad()
                    score_loss = torch.mean(-log_probs * Ls)  # - derivative of Score function，Ls是两者之和
                    entropy_penalty = torch.mean(entropies)  # Entropy penalty
                    controller_loss = score_loss - 1e-5 * entropy_penalty
                    # print(score_loss, entropy_penalty, controller_loss)
                    controller_loss.backward()
                    # controller.module.print_grad()
                    controller_optimizer.step()
                    # cost_time[5] += time.time()

                # print("end controller update")
                model.train()

            # -------------------------------日志打印和模型存储 -------------------------------------------------------
            # log print
            if total_steps % opt.display_freq == 0: # iter计算
                visualizer.display_current_results(model.get_current_visuals(), epoch, int(i / 100))

            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time)
                iter_start_time = time.time()
                visualizer.print_current_errors(epoch, total_steps, model.get_current_losses(), t)
                # print(cost_time)
                # cost_time = [0 for _ in range(6)]

            if ((i + 1) % opt.valid_freq == 0 or i == len(dataset) - 1) and opt.valid != 0:
                _, psnr_v, _, psnr_v_m = valid_metrics_cal(valid_dataset, v_l, model, visualizer, epoch, False, m_type=1)
                _, psnr_t, _, psnr_t_m = valid_metrics_cal(test_dataset, t_l, model, visualizer, epoch, False, m_type=1)
                if i == len(dataset) - 1:
                    visualizer.print_valid_metric_list(epoch, [psnr_v, psnr_t, psnr_v_m, psnr_t_m])
                else:
                    visualizer.print_valid_metric_list(epoch, [psnr_v, psnr_t, psnr_v_m, psnr_t_m], i)
                if i == len(dataset) - 1:
                    if "ens" in opt.dataset_mode:
                        interal = 500
                    else:
                        interal = 50
                    if epoch > begin_epoch + interal:
                        stop_epoch += 1

                if psnr_v_m > best_metrics:
                    best_metrics = psnr_v_m
                    stop_epoch = 0
                    save_path = os.path.join(model.save_dir, 'best_net_Controller.pth')
                    torch.save(controller.state_dict(), save_path)
                    model.save_networks('best')

        # model save
        save_path = os.path.join(model.save_dir, 'latest_net_Controller.pth')
        torch.save(controller.state_dict(), save_path)
        model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0: # 每隔多少轮保存一次
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks(epoch)
            save_path = os.path.join(model.save_dir, '%s_net_Controller.pth' % epoch)
            torch.save(controller.state_dict(), save_path)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        if stop_epoch > 4:
            return


if __name__ == "__main__":
    main()
