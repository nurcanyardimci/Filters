from numpy import array

def bresenham_indexes(A, B):
    path = []
    (x0, y0) = (A.x, A.y)
    (x1, y1) = (B.x, B.y)
    kx = -1 if x0 > x1 else 1
    ky = -1 if y0 > y1 else 1
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    (x, y) = (x0, y0)
    if dx > dy:
        e = dx / 2.0
        while x != x1:
            path.append([x, y])
            e -= dy
            if e < 0:
                y += ky
                e += dx
            x += kx
    else:
        e = dy / 2.0
        while y != y1:
            path.append([x, y])
            e -= dx
            if e < 0:
                x += kx
                e += dy
            y += ky
    path.append([x, y])
    return array(path)
