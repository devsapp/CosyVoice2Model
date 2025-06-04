
> 注：当前项目为 Serverless Devs 应用，由于应用中会存在需要初始化才可运行的变量（例如应用部署地区、函数名等等），所以**不推荐**直接 Clone 本仓库到本地进行部署或直接复制 s.yaml 使用，**强烈推荐**通过 `s init ${模版名称}` 的方法或应用中心进行初始化，详情可参考[部署 & 体验](#部署--体验) 。

# CosyVoice2 帮助文档

<description>

部署 CosyVoice2 到 FunctionAI

</description>


## 资源准备

使用该项目，您需要有开通以下服务并拥有对应权限：

<service>



| 服务/业务 |  权限  | 相关文档 |
| --- |  --- | --- |
| 函数计算 |  AliyunFCFullAccess | [帮助文档](https://help.aliyun.com/product/2508973.html) [计费文档](https://help.aliyun.com/document_detail/2512928.html) |

</service>

<remark>



</remark>

<disclaimers>



</disclaimers>

## 部署 & 体验

<appcenter>
   
- :fire: 通过 [云原生应用开发平台 CAP](https://cap.console.aliyun.com/template-detail?template=CosyVoice2) ，[![Deploy with Severless Devs](https://img.alicdn.com/imgextra/i1/O1CN01w5RFbX1v45s8TIXPz_!!6000000006118-55-tps-95-28.svg)](https://cap.console.aliyun.com/template-detail?template=CosyVoice2) 该应用。
   
</appcenter>
<deploy>
    
   
</deploy>

## 案例介绍

<appdetail id="flushContent">

本应用旨在帮助开发者实现将 CosyVoice2 应用部署到阿里云函数计算 FunctionAI，并且轻松实现一键拉起 CosyVoice2 开源应用，实时体验到语音合成效果

## 部署准备

您需要先开通以下服务：

- 函数计算 FC：创建 AliyunFCDefaultRole 角色，点击同意授权![](https://img.alicdn.com/imgextra/i3/O1CN01FIQqTU1fODh2Z4r1v_!!6000000003996-0-tps-3024-1646.jpg)![](https://img.alicdn.com/imgextra/i3/O1CN0117ZZYx1iv90r8kQEj_!!6000000004474-0-tps-3024-1646.jpg)


- 日志服务SLS：请到[SLS控制台](https://sls.console.aliyun.com/lognext/profile)开通日志服务

</appdetail>




## 架构图

<framework id="flushContent">

![](https://img.alicdn.com/imgextra/i2/O1CN01vQcpLu1Zb9DnAV6Og_!!6000000003212-2-tps-804-348.png)

</framework>

## 使用流程

<usedetail id="flushContent">

**步骤一**：登录阿里云函数计算控制台，点击左侧「FunctionAI」，在「人工智能」页签下找到开源托管 CosyVoice2-0.5 模型，点击并「立即部署」 ![](https://img.alicdn.com/imgextra/i2/O1CN01YsdL5Z1FuZhr4qEOd_!!6000000000547-2-tps-493-165.png)


**步骤二**：自行选择部署的地域和卡型和 OSS 存储桶，然后点击部署项目 ![](https://img.alicdn.com/imgextra/i4/O1CN01s2SnoP1mgj3AqvLli_!!6000000004984-2-tps-1413-1222.png)


**步骤三**：耐心等待，显示部署成功后，即可看到模型 API ![](https://img.alicdn.com/imgextra/i1/O1CN01pmwKyH1I68wRfWDu6_!!6000000000843-2-tps-2165-885.png)

</usedetail>




## 注意事项

<matters id="flushContent">

</matters>



