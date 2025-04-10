from fontTools import ttLib
import numpy as np

from Tools.util.ttf.bezier import multi_raster, raster
import torch

from Tools.util.utils import vutils
# 定义循环左移函数
def circular_shift_left(array, shift):
    shift %= len(array)  # 处理超出数组长度的位移量
    return np.concatenate((array[shift:], array[:shift]))

class TTFManager:
    def __init__(self, ttf_path) -> None:
        """TTF上下文管理器

        TTF详情可参考：
            https://www.modb.pro/db/175187
            https://www.modb.pro/db/175186
        
        Tables:
            cmap: 字符到字形的映射
            glyf: 字形轮廓数据
            head: 字体全局信息
            hhead: 水平布局字体的通用信息
            hmtx: 每个字符的水平布局度量
            loca: 每个字形在glyf中的偏移
        
        Args:
            ttf_path (_type_): _description_
        """
        self.ttf = ttLib.TTFont(ttf_path)
        self.glyf = self.ttf["glyf"]
        self.glyph_set = self.ttf.getGlyphSet()
        self.chars = set(self.ttf.getGlyphNames())
        
    
    
    def getCmap(self):
        """返回最佳的cmap表，cmap是字符到字形的映射
        Returns:
            _type_: _description_
        """
        return self.ttf.getBestCmap()
    
    def getChars(self):
        """返回TTF中存储的所有字符

        Returns:
            _type_: _description_
        """
        return self.chars
    
    def getCharId(self, char):
        if '\u4e00' <= char <= '\u9fff':
            # 判断它到底在不在里面
            id = "uni" + hex(ord(char))[2:].upper()
            if id in self.chars:
                return id
            id = hex(ord(char))[2:].upper()
            if id in self.chars:
                return id
            id = "uni" + hex(ord(char))[2:].lower()
            if id in self.chars:
                return id
            id = hex(ord(char))[2:].lower()
            if id in self.chars:
                return id
            return None
            pass
        else:
            return char
        pass
    
    def getBezier(self, id):
        """获取字符的贝塞尔曲线表示

        Args:
            id (_type_): _description_ 字符id
        Return:
            n个curve，其中每个curve由points和flag组成
            points: 控制点的坐标
            flags: 控制点的类型
        """
        data = self.glyf[id]
        coordinates = data.coordinates
        flags = data.flags
        ends = data.endPtsOfContours
        coordinates = np.array(list(coordinates))
        flags = np.array([int(bit) for bit in flags])    
        print(ends)
        curves = []
        start = 0
        for end in ends:
            coordinate = coordinates[start:end+1, :]
            flag = flags[start:end+1]
            start = end+1
            
            if flag.shape[0] == 0:
                continue
            
            if flag[0] == 0:
                coordinate = circular_shift_left(coordinate, 1)
                flag = circular_shift_left(flag, 1)
            
            offset = 0
            for idx in range(flag.shape[0]):
                nxt_idx = (idx + 1) % flag.shape[0]
                if flag[idx] == 1 and flag[nxt_idx] == 1:
                    flag = np.insert(flag, idx+1, 0)
                    new_data = (coordinate[idx]+coordinate[nxt_idx]) / 2
                    new_data = new_data.reshape(1, 2) 
                    coordinate = np.insert(coordinate, idx+1, new_data, axis=0)
                    pass
                else:
                    
                    pass
                pass
            curves.append((
                coordinate,
                flag
            ))
            pass
        return curves
        pass
    
    def getImg(self, id):
        
        pass
    
    pass

if __name__ == "__main__":
    manager = TTFManager("/home/luyx/env_sn/workspace/diffusion/kai.ttf")
    id = manager.getCharId("天")
    print(id)
    # curves = manager.getBezier('二')
    # c = []
    # print(len(curves))
    # for curve in curves:
    #     points, flags = curve
    #     points = torch.from_numpy(points)
    #     c.append(points)
    #     pass
    # img, sdf = multi_raster(c, 280, 256)
    # vutils().save_image(img.unsqueeze(0), "./test.png")
    #img, sdf = raster(points, 256, 256)
    #print(img.size())
    #vutils().save_image(img.unsqueeze(0), "./test.png")
    pass