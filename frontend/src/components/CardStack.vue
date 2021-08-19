<template>
  <div :class="classObjectUpper">
    <div v-for="(card, idx) in cards" :key="idx" :class="classObject">
      <p>CARD: {{ card.name }}</p>
      <Card
        :alt="alt"
        :card="card.img"
        :suite="card.name"
        :height="height"
        :width="width"
      />
    </div>
  </div>
</template>

<script>
import Card from "./Card.vue";

export default {
  components: { Card },
  props: {
    position: {
      type: String,
      required: true,
    },
    height: {
      type: String,
      default: '100',
      required: false,
    },
    width: {
      type: String,
      default: '100',
      required: false,
    },
    // cards: Array,
  },
  data: () => {
    return {
      alt: "some alt",
      // position: "left", // up, down, left, right
      cards: [
        {
          name: "club",
          img: "2",
        },
        {
          name: "club",
          img: "3",
        },
        {
          name: "spade",
          img: "4",
        },
      ],
    };
  },
  computed: {
    classObjectUpper: function () {
      if (this.position === "left" || this.position === "right") {
        return {
          "col-2": true,
        };
      } else {
        return {
          row: true,
        };
      }
    },
    classObject: function () {
      if (this.position === "left" || this.position === "right") {
        return {
          row: true,
          col: false,
        };
      } else {
        return {
          row: false,
          col: true,
        };
      }
    },
  },
};
</script>
<style scoped>
</style>