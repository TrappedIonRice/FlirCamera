if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    import laserbeamsize as lbs
    beam = imageio.imread("test_image.jpg")
    x, y, dx, dy, phi = lbs.beam_size(beam)
    print("The center of the beam ellipse is at (%.0f, %.0f)" % (x, y))
    print("The ellipse diameter (closest to horizontal) is %.0f pixels" % dx)
    print("The ellipse diameter (closest to   vertical) is %.0f pixels" % dy)
    print("The ellipse is rotated %.0f° ccw from horizontal" % (phi * 180 / 3.1416))
    lbs.beam_size_plot(beam)
    plt.show()