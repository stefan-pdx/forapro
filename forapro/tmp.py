plt.figure(figsize=(5,5))
for i,m in enumerate(masks):
  plt.clf()
  plt.quiver(m.real, m.imag)
  plt.title("Circular Kernel %i"%(i+1))
  plt.savefig("circular-kernel-%i.png"%(i+1))
