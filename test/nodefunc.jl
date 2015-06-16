nn = S.NN([19,1], [S.Relu()])
ref = S.calc(nn, vec(vars[12345,:]))
ts = S.cycle(nn, vec(vars[12345,:]), [res[12456]])
nn.ws[1][1,1] += 1e-8
dref = ( (S.calc(nn, vec(vars[12345,:]))-[res[12456]]).^2 - (ref-[res[12456]]).^2)*1e8  # 1.536
[ ts.dws[1][1,1] dref ]


nn = S.NN([19,1], [S.SoftRelu()])
ref = S.calc(nn, vec(vars[12345,:]))
ts = S.cycle(nn, vec(vars[12345,:]), [res[12456]])
nn.ws[1][1,1] += 1e-8
dref = ( (S.calc(nn, vec(vars[12345,:]))-[res[12456]]).^2 - (ref-[res[12456]]).^2)*1e8  # 1.536
[ ts.dws[1][1,1] dref ]

nn = S.NN([19,1], [S.Sigmoid()])
ref = S.calc(nn, vec(vars[12345,:]))
ts = S.cycle(nn, vec(vars[12345,:]), [res[12456]])
nn.ws[1][1,1] += 1e-8
dref = ( (S.calc(nn, vec(vars[12345,:]))-[res[12456]]).^2 - (ref-[res[12456]]).^2)*1e8  # 1.536
[ ts.dws[1][1,1] dref ]  # faux

nn = S.NN([19,1], [S.Tanh()])
ref = S.calc(nn, vec(vars[12345,:]))
ts = S.cycle(nn, vec(vars[12345,:]), [res[12456]])
nn.ws[1][1,1] += 1e-8
dref = ( (S.calc(nn, vec(vars[12345,:]))-[res[12456]]).^2 - (ref-[res[12456]]).^2)*1e8  # 1.536
[ ts.dws[1][1,1] dref ]
