from Network import Network


def main():
    network = Network([3, 2, 2, 1])
    network.print_network()
    network.set_inputs([[1.], [0.], [1.]])
    network.forward_propagation()
    print()
    print()
    print()
    print(network.get_outputs())


if __name__ == "__main__":
    main()
